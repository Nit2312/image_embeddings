/**
 * Cloudflare Worker for Image Vector Store Management
 * 
 * Endpoints:
 * 1. POST /create-index - Create/generate vector store (index)
 * 2. POST /add-vector - Add a vector to existing vector store
 * 3. DELETE /delete-vector/:id - Delete an image vector from vector store
 * 4. POST /search - Search: receives image, converts to vector, similarity search, returns top 20
 */

interface Env {
  VECTORIZE: Vectorize;
  ACCOUNT_ID: string;
  API_TOKEN: string;
  VECTORIZE_INDEX: string;
  EMBEDDING_API_URL?: string; // Optional: URL to external embedding service
}

interface VectorMetadata {
  filename?: string;
  path?: string;
  uploaded_at?: string;
  source?: string;
  product_id?: string;
  content_type?: string;
  [key: string]: any;
}

interface UpsertVector {
  id: string;
  values: number[];
  metadata?: VectorMetadata;
}

interface SearchRequest {
  image?: string; // base64 encoded image
  imageUrl?: string; // URL to image
  topK?: number; // number of results (default 20)
  filter?: Record<string, any>; // metadata filter
}

interface CreateIndexRequest {
  dimensions: number;
  metric?: 'cosine' | 'euclidean';
  description?: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    // Handle CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      // 1. Create/Generate Vector Store (Index)
      if (path === '/create-index' && method === 'POST') {
        return await handleCreateIndex(request, env, corsHeaders);
      }

      // 2. Add Vector to Vector Store
      if (path === '/add-vector' && method === 'POST') {
        return await handleAddVector(request, env, corsHeaders);
      }

      // 3. Delete Vector from Vector Store
      if (path.startsWith('/delete-vector/') && method === 'DELETE') {
        const id = path.split('/delete-vector/')[1];
        return await handleDeleteVector(id, env, corsHeaders);
      }

      // 4. Search - Image similarity search
      if (path === '/search' && method === 'POST') {
        return await handleSearch(request, env, corsHeaders);
      }

      // Health check
      if (path === '/health' && method === 'GET') {
        return new Response(JSON.stringify({ status: 'ok', service: 'vector-store-worker' }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }

      return new Response('Not Found', { 
        status: 404,
        headers: corsHeaders 
      });
    } catch (error: any) {
      console.error('Error:', error);
      return new Response(JSON.stringify({ 
        success: false,
        error: error.message || 'Internal Server Error',
      }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  },
};

function json(
  data: unknown,
  corsHeaders: Record<string, string>,
  status = 200
): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

function badRequest(corsHeaders: Record<string, string>, message: string): Response {
  return json({ success: false, error: message }, corsHeaders, 400);
}

/**
 * 1. Create/Generate Vector Store (Index)
 */
async function handleCreateIndex(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const body: CreateIndexRequest = await request.json();
    const { dimensions, metric = 'cosine', description } = body;

    if (!dimensions || dimensions < 1) {
      return badRequest(corsHeaders, 'dimensions is required and must be > 0');
    }

    // Validate environment variables
    if (!env.ACCOUNT_ID || !env.VECTORIZE_INDEX || !env.API_TOKEN) {
      return json(
        {
          success: false,
          error:
            'Missing required environment variables. Please set ACCOUNT_ID, VECTORIZE_INDEX, and API_TOKEN secrets in your worker.',
        },
        corsHeaders,
        500
      );
    }

    // Create index via Cloudflare API
    const url = `https://api.cloudflare.com/client/v4/accounts/${env.ACCOUNT_ID}/vectorize/v2/indexes/${env.VECTORIZE_INDEX}`;
    
    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${env.API_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        dimensions,
        metric,
        description: description || `Image vector store created via worker`,
      }),
    });

    // Check if response is JSON before parsing
    const contentType = response.headers.get('content-type') || '';
    let result: any;
    
    if (contentType.includes('application/json')) {
      try {
        const text = await response.text();
        result = JSON.parse(text.trim());
      } catch (parseError: any) {
        return new Response(JSON.stringify({
          success: false,
          error: `Invalid JSON response from Cloudflare API`,
        }), {
          status: response.status,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }
    } else {
      const text = await response.text();
      return new Response(JSON.stringify({
        success: false,
        error: `Unexpected response format: ${text.substring(0, 200)}`,
      }), {
        status: response.status,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    if (response.ok && result.success) {
      return json(
        {
          success: true,
          message: 'Vector store created successfully',
          index: result.result,
        },
        corsHeaders
      );
    }
    return json(
      {
        success: false,
        error: result.errors || 'Failed to create index',
      },
      corsHeaders,
      response.status
    );
  } catch (error: any) {
    return json({ success: false, error: error.message }, corsHeaders, 500);
  }
}

/**
 * 2. Add Vector to Vector Store
 */
async function handleAddVector(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const contentType = request.headers.get('content-type') || '';
    
    let imageBuffer: ArrayBuffer;
    let metadata: VectorMetadata = {};
    let vectorId: string;

    if (contentType.includes('multipart/form-data')) {
      // Handle multipart form data (file upload)
      const formData = await request.formData();
      const file = formData.get('image') as File;
      const id = formData.get('id') as string;
      const metadataStr = formData.get('metadata') as string;

      if (!file) {
        return badRequest(corsHeaders, 'No image file provided');
      }

      imageBuffer = await file.arrayBuffer();
      vectorId = id || crypto.randomUUID();
      
      if (metadataStr) {
        try {
          metadata = JSON.parse(metadataStr);
        } catch (e) {
          // Ignore parse errors
        }
      }

      metadata.filename = file.name;
      metadata.uploaded_at = new Date().toISOString();
      metadata.source = metadata.source || 'upload';
    } else if (contentType.includes('application/json')) {
      // Handle JSON with base64 image or image URL
      const body = await request.json();
      const { image, imageUrl, id, metadata: customMetadata } = body;

      vectorId = id || crypto.randomUUID();

      if (image) {
        // Base64 encoded image
        const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        imageBuffer = bytes.buffer;
      } else if (imageUrl) {
        // Fetch image from URL
        const imageResponse = await fetch(imageUrl);
        if (!imageResponse.ok) {
          return new Response(JSON.stringify({ error: 'Failed to fetch image from URL' }), {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          });
        }
        imageBuffer = await imageResponse.arrayBuffer();
      } else {
        return badRequest(corsHeaders, 'No image provided (use image or imageUrl)');
      }

      if (customMetadata) {
        metadata = { ...metadata, ...customMetadata };
      }
      metadata.uploaded_at = new Date().toISOString();
      metadata.source = metadata.source || (imageUrl ? 'url' : 'base64');
    } else {
      return json({ success: false, error: 'Unsupported content type' }, corsHeaders, 415);
    }

    // Convert image to vector
    const vector = await convertImageToVector(imageBuffer, env);
    if (!Array.isArray(vector) || vector.length < 1) {
      return json({ success: false, error: 'Embedding service returned an invalid vector' }, corsHeaders, 502);
    }

    // Upsert into Vectorize
    const upsertResult = await env.VECTORIZE.upsert([{
      id: vectorId,
      values: vector,
      metadata,
    }]);

    return json(
      {
        success: true,
        id: vectorId,
        mutationId: upsertResult.mutationId,
        message: 'Vector added successfully',
      },
      corsHeaders
    );
  } catch (error: any) {
    return json({ success: false, error: error.message }, corsHeaders, 500);
  }
}

/**
 * 3. Delete Vector from Vector Store
 */
async function handleDeleteVector(
  id: string,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    if (!id) {
      return badRequest(corsHeaders, 'Vector ID is required');
    }

    const mutation = await env.VECTORIZE.deleteByIds([id]);
    return json(
      {
        success: true,
        id,
        mutationId: mutation.mutationId,
        message: 'Vector delete requested',
      },
      corsHeaders
    );
  } catch (error: any) {
    return json({ success: false, error: error.message || 'Internal server error' }, corsHeaders, 500);
  }
}

/**
 * 4. Search - Image similarity search
 */
async function handleSearch(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const body: SearchRequest = await request.json();

    let imageBuffer: ArrayBuffer;

    if (body.image) {
      // Base64 encoded image
      const base64Data = body.image.replace(/^data:image\/\w+;base64,/, '');
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      imageBuffer = bytes.buffer;
    } else if (body.imageUrl) {
      // Fetch image from URL
      const imageResponse = await fetch(body.imageUrl);
      if (!imageResponse.ok) {
        return new Response(JSON.stringify({ error: 'Failed to fetch image from URL' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }
      imageBuffer = await imageResponse.arrayBuffer();
    } else {
      return badRequest(corsHeaders, 'No image provided (use image or imageUrl)');
    }

    // Convert image to vector
    const queryVector = await convertImageToVector(imageBuffer, env);
    if (!Array.isArray(queryVector) || queryVector.length < 1) {
      return json({ success: false, error: 'Embedding service returned an invalid vector' }, corsHeaders, 502);
    }

    // Perform similarity search
    const topK = body.topK || 20;
    const matches = await env.VECTORIZE.query(queryVector, {
      topK,
      returnValues: false, // Don't return full vectors to save bandwidth
      returnMetadata: true,
      filter: body.filter,
    });

    return json(
      {
        success: true,
        matches: matches.matches || [],
        count: matches.matches?.length || 0,
      },
      corsHeaders
    );
  } catch (error: any) {
    return json({ success: false, error: error.message }, corsHeaders, 500);
  }
}

/**
 * Convert image to vector using DINOv3
 * This function calls an external embedding service or uses Workers AI
 */
async function convertImageToVector(
  imageBuffer: ArrayBuffer,
  env: Env
): Promise<number[]> {
  // Option 1: Use external embedding API (if EMBEDDING_API_URL is set)
  if (env.EMBEDDING_API_URL) {
    // Check if URL is localhost (won't work from Cloudflare)
    if (env.EMBEDDING_API_URL.includes('localhost') || env.EMBEDDING_API_URL.includes('127.0.0.1')) {
      throw new Error(
        'Embedding API URL cannot be localhost. The worker runs on Cloudflare and cannot access localhost. ' +
        'Please deploy your embedding API to a public URL (Railway, Render, Fly.io) or use a tunnel service like ngrok.'
      );
    }

    let lastError: Error | null = null;

    // Try /embed-binary endpoint first (for raw binary)
    try {
      const response = await fetch(`${env.EMBEDDING_API_URL}/embed-binary`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
        },
        body: imageBuffer,
      });

      if (response.ok) {
        const result = await response.json();
        const embedding = result.embedding || result.vector || result.values;
        if (embedding && Array.isArray(embedding)) {
          return embedding;
        }
      } else {
        const errorText = await response.text().catch(() => response.statusText);
        lastError = new Error(`HTTP ${response.status}: ${errorText}`);
      }
    } catch (e: any) {
      lastError = e;
      console.warn('Failed to use /embed-binary, trying /embed endpoint:', e.message);
    }

    // Fallback to /embed endpoint (multipart form)
    try {
      const formData = new FormData();
      const blob = new Blob([imageBuffer]);
      formData.append('file', blob, 'image.jpg');

      const response = await fetch(`${env.EMBEDDING_API_URL}/embed`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => response.statusText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      const embedding = result.embedding || result.vector || result.values;
      if (embedding && Array.isArray(embedding)) {
        return embedding;
      }
      throw new Error('Invalid response format from embedding API');
    } catch (e: any) {
      throw new Error(`Embedding API error: ${e.message || 'Unknown error'}. URL: ${env.EMBEDDING_API_URL}`);
    }
  }

  // Option 2: Use Cloudflare Workers AI (if available)
  // Note: Cloudflare Workers AI may not have DINOv3, but you can use other models
  // This is a placeholder - you'll need to implement based on available AI models
  
  // Option 3: Call back to your Python service
  // You could deploy your Python embedding service and call it from here
  
  throw new Error(
    'No embedding service configured. Set EMBEDDING_API_URL environment variable ' +
    'or implement Workers AI integration.'
  );
}
