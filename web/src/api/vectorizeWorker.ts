export type WorkerMatch = {
  id: string
  score: number
  metadata?: Record<string, any>
}

type SearchResponse = {
  success: boolean
  matches: WorkerMatch[]
  count: number
  error?: string
}

type AddResponse = {
  success: boolean
  id: string
  mutationId: string
  message?: string
  error?: string
}

type DeleteResponse = {
  success: boolean
  id: string
  mutationId: string
  message?: string
  error?: string
}

function workerBaseUrl(): string {
  const raw = (import.meta.env.VITE_WORKER_URL as string | undefined) ?? ''
  return raw.replace(/\/+$/, '')
}

async function readJsonOrThrow<T>(res: Response): Promise<T> {
  const text = await res.text()
  try {
    return JSON.parse(text.trim()) as T
  } catch {
    throw new Error(`Non-JSON response (HTTP ${res.status}): ${text.slice(0, 200)}`)
  }
}

export async function searchByImage(args: {
  file: File
  topK: number
  filter?: Record<string, any>
}): Promise<{ matches: WorkerMatch[]; count: number }> {
  const base64 = await fileToDataUrl(args.file)
  const res = await fetch(`${workerBaseUrl()}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: base64,
      topK: args.topK,
      filter: args.filter,
    }),
  })
  const data = await readJsonOrThrow<SearchResponse>(res)
  if (!res.ok || !data.success) throw new Error(data.error || `Search failed (HTTP ${res.status})`)
  return { matches: data.matches ?? [], count: data.count ?? 0 }
}

export async function addVector(args: {
  file: File
  id?: string
  metadata?: Record<string, any>
}): Promise<{ id: string; mutationId: string }> {
  const form = new FormData()
  form.append('image', args.file, args.file.name)
  if (args.id) form.append('id', args.id)
  if (args.metadata) form.append('metadata', JSON.stringify(args.metadata))

  const res = await fetch(`${workerBaseUrl()}/add-vector`, { method: 'POST', body: form })
  const data = await readJsonOrThrow<AddResponse>(res)
  if (!res.ok || !data.success) throw new Error(data.error || `Add failed (HTTP ${res.status})`)
  return { id: data.id, mutationId: data.mutationId }
}

export async function deleteVector(args: { id: string }): Promise<{ id: string; mutationId: string }> {
  const res = await fetch(`${workerBaseUrl()}/delete-vector/${encodeURIComponent(args.id)}`, {
    method: 'DELETE',
  })
  const data = await readJsonOrThrow<DeleteResponse>(res)
  if (!res.ok || !data.success) throw new Error(data.error || `Delete failed (HTTP ${res.status})`)
  return { id: data.id, mutationId: data.mutationId }
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('Failed reading file'))
    reader.readAsDataURL(file)
  })
}

