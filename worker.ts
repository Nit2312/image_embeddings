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
        error: error.message || 'Internal Server Error',
        details: error.stack 
      }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  },
};

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
      return new Response(JSON.stringify({ error: 'dimensions is required and must be > 0' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Validate environment variables
    if (!env.ACCOUNT_ID || !env.VECTORIZE_INDEX || !env.API_TOKEN) {
      return new Response(JSON.stringify({ 
        success: false,
        error: 'Missing required environment variables. Please set ACCOUNT_ID, VECTORIZE_INDEX, and API_TOKEN secrets in your worker.' 
      }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
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
      return new Response(JSON.stringify({
        success: true,
        message: 'Vector store created successfully',
        index: result.result,
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    } else {
      return new Response(JSON.stringify({
        success: false,
        error: result.errors || 'Failed to create index',
      }), {
        status: response.status,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
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
        return new Response(JSON.stringify({ error: 'No image file provided' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
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
        return new Response(JSON.stringify({ error: 'No image provided (use image or imageUrl)' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }

      if (customMetadata) {
        metadata = { ...metadata, ...customMetadata };
      }
      metadata.uploaded_at = new Date().toISOString();
    } else {
      return new Response(JSON.stringify({ error: 'Unsupported content type' }), {
        status: 415,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Convert image to vector
    const vector = await convertImageToVector(imageBuffer, env);

    // Upsert into Vectorize
    const upsertResult = await env.VECTORIZE.upsert([{
      id: vectorId,
      values: vector,
      metadata,
    }]);

    return new Response(JSON.stringify({
      success: true,
      id: vectorId,
      mutationId: upsertResult.mutationId,
      message: 'Vector added successfully',
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
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
      return new Response(JSON.stringify({ error: 'Vector ID is required' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Validate environment variables
    if (!env.ACCOUNT_ID || !env.VECTORIZE_INDEX || !env.API_TOKEN) {
      return new Response(JSON.stringify({ 
        success: false,
        error: 'Missing required environment variables. Please set ACCOUNT_ID, VECTORIZE_INDEX, and API_TOKEN secrets in your worker.' 
      }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Cloudflare Vectorize delete endpoint
    // Note: Vectorize API might use different endpoint format
    const url = `https://api.cloudflare.com/client/v4/accounts/${env.ACCOUNT_ID}/vectorize/v2/indexes/${env.VECTORIZE_INDEX}/delete`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.API_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ids: [id],
      }),
    });

    // Handle 404 - endpoint might not exist or Vectorize doesn't support delete
    if (response.status === 404) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Delete endpoint not found. Cloudflare Vectorize may not support deleting individual vectors via API. Consider using the Vectorize binding or recreating the index without the vector.',
        statusCode: 404,
        note: 'Vectorize delete functionality may require using wrangler CLI or may not be available via API',
      }), {
        status: 404,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Check if response is JSON before parsing
    const contentType = response.headers.get('content-type') || '';
    let result: any;
    
    if (contentType.includes('application/json')) {
      try {
        const text = await response.text();
        result = JSON.parse(text.trim());
      } catch (parseError: any) {
        // If JSON parsing fails, get text response
        const text = await response.text();
        return new Response(JSON.stringify({
          success: false,
          error: `Invalid JSON response: ${text.substring(0, 200)}`,
          statusCode: response.status,
        }), {
          status: response.status,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }
    } else {
      // Response is not JSON, get text
      const text = await response.text();
      return new Response(JSON.stringify({
        success: false,
        error: `Unexpected response format (${response.status}): ${text.substring(0, 200)}`,
        statusCode: response.status,
      }), {
        status: response.status,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    if (response.ok && result.success) {
      return new Response(JSON.stringify({
        success: true,
        message: 'Vector deleted successfully',
        id,
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    } else {
      // Extract error message properly with more details
      let errorMsg = 'Failed to delete vector';
      
      if (response.status === 401 || response.status === 403) {
        errorMsg = 'Authentication failed. Please check your API_TOKEN. Make sure it has Vectorize permissions and is not expired.';
      } else if (result.errors && Array.isArray(result.errors)) {
        errorMsg = result.errors.map((e: any) => {
          if (e.message) return e.message;
          if (e.code) return `Error ${e.code}: ${e.message || 'Unknown error'}`;
          return JSON.stringify(e);
        }).join(', ');
      } else if (result.error) {
        errorMsg = typeof result.error === 'string' ? result.error : JSON.stringify(result.error);
      } else if (result.message) {
        errorMsg = result.message;
      }
      
      // Add status code info
      if (response.status === 401) {
        errorMsg = `Authentication failed (401): ${errorMsg}. Please verify your API_TOKEN has Vectorize:Edit permissions.`;
      } else if (response.status === 403) {
        errorMsg = `Permission denied (403): ${errorMsg}. Your API_TOKEN may not have the required permissions.`;
      } else if (response.status === 404) {
        errorMsg = `Not found (404): ${errorMsg}. Check if the index name '${env.VECTORIZE_INDEX}' is correct.`;
      }
      
      return new Response(JSON.stringify({
        success: false,
        error: errorMsg,
        statusCode: response.status,
      }), {
        status: response.status,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  } catch (error: any) {
    return new Response(JSON.stringify({ 
      success: false,
      error: error.message || 'Internal server error' 
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
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
    const contentType = request.headers.get('content-type') || '';
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
      return new Response(JSON.stringify({ error: 'No image provided (use image or imageUrl)' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Convert image to vector
    const queryVector = await convertImageToVector(imageBuffer, env);

    // Perform similarity search
    const topK = body.topK || 20;
    const matches = await env.VECTORIZE.query(queryVector, {
      topK,
      returnValues: false, // Don't return full vectors to save bandwidth
      returnMetadata: true,
      filter: body.filter,
    });

    return new Response(JSON.stringify({
      success: true,
      matches: matches.matches || [],
      count: matches.matches?.length || 0,
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
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
