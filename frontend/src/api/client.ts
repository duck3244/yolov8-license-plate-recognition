import axios from 'axios'

// dev: Vite proxy(/api) → http://127.0.0.1:8000
// prod: FastAPI가 같은 origin으로 정적 SPA를 서빙하므로 그대로 /api
export const api = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

export interface DetectResponse {
  success: boolean
  detection_id?: number
  plate_number?: string
  confidence?: number
  processing_time: number
  result_image_url?: string
  error?: string
}

export interface DetectionItem {
  id: number
  plate_number: string
  confidence: number
  timestamp: string
  original_filename?: string | null
  processing_time?: number | null
  result_image_url?: string | null
}

export async function uploadImage(file: File): Promise<DetectResponse> {
  const fd = new FormData()
  fd.append('image', file)
  const { data } = await api.post<DetectResponse>('/detect', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function fetchHistory(limit = 20): Promise<DetectionItem[]> {
  const { data } = await api.get<{ detections: DetectionItem[] }>('/history', {
    params: { limit },
  })
  return data.detections
}
