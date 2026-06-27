import axios from 'axios'

const aiApi = axios.create({
  baseURL: '/ai',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface ReviewPrecheckRequest {
  content: string
  question: string
  n_results: number
}

export interface ReviewPrecheckResponse {
  score: number
  approvalRisk: 'LOW' | 'MEDIUM' | 'HIGH'
  reasons: string[]
  suggestions: string[]
  similarDocuments: Array<{
    content: string
    metadata: Record<string, unknown>
    distance: number | null
  }>
  sentiment: Record<string, unknown>
  qa: {
    answer: string
    score: number
    start: number
    end: number
  }
  metrics: {
    totalMs: number
    searchMs: number
    sentimentMs: number
    qaMs: number
    cacheHit: boolean
  }
}

export async function precheckReview(request: ReviewPrecheckRequest) {
  const response = await aiApi.post<ReviewPrecheckResponse>('/review/precheck', request)
  return response.data
}
