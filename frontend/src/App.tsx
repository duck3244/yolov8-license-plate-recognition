import { useCallback, useEffect, useState } from 'react'
import { Upload, Loader2, Car, History as HistoryIcon } from 'lucide-react'
import {
  DetectResponse,
  DetectionItem,
  fetchHistory,
  uploadImage,
} from './api/client'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<DetectResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<DetectionItem[]>([])

  // 객체 URL 정리
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  const loadHistory = useCallback(async () => {
    try {
      const items = await fetchHistory(10)
      setHistory(items)
    } catch (e) {
      console.error(e)
    }
  }, [])

  useEffect(() => {
    void loadHistory()
  }, [loadHistory])

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null
    if (preview) URL.revokeObjectURL(preview)
    setFile(f)
    setPreview(f ? URL.createObjectURL(f) : null)
    setResult(null)
    setError(null)
  }

  const onSubmit = async () => {
    if (!file) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const res = await uploadImage(file)
      setResult(res)
      void loadHistory()
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? '요청 실패')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-full">
      <header className="bg-slate-900 text-white shadow">
        <div className="mx-auto max-w-5xl px-6 py-4 flex items-center gap-3">
          <Car className="w-6 h-6" />
          <h1 className="text-xl font-semibold">번호판 인식</h1>
          <span className="text-xs text-slate-400 ml-2">MVP v3.0</span>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8 space-y-8">
        <section className="bg-white rounded-2xl shadow p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5" />
            이미지 업로드
          </h2>
          <input
            type="file"
            accept="image/jpeg,image/png,image/bmp,image/tiff"
            onChange={onFileChange}
            className="block w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-slate-900 file:text-white file:px-4 file:py-2 file:text-sm hover:file:bg-slate-700 cursor-pointer"
          />

          {preview && (
            <div className="mt-4 grid md:grid-cols-2 gap-4">
              <figure>
                <figcaption className="text-sm text-slate-500 mb-1">미리보기</figcaption>
                <img
                  src={preview}
                  alt="upload preview"
                  className="rounded-lg border border-slate-200 max-h-72 object-contain bg-slate-50"
                />
              </figure>
              {result?.success && result.result_image_url && (
                <figure>
                  <figcaption className="text-sm text-slate-500 mb-1">인식 결과</figcaption>
                  <img
                    src={result.result_image_url}
                    alt="result"
                    className="rounded-lg border border-slate-200 max-h-72 object-contain bg-slate-50"
                  />
                </figure>
              )}
            </div>
          )}

          <button
            onClick={onSubmit}
            disabled={!file || busy}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-slate-900 text-white px-5 py-2.5 text-sm font-medium hover:bg-slate-700 disabled:bg-slate-300 disabled:cursor-not-allowed"
          >
            {busy && <Loader2 className="w-4 h-4 animate-spin" />}
            {busy ? '인식 중...' : '번호판 인식'}
          </button>

          {error && (
            <p className="mt-3 text-sm text-red-600">⚠️ {error}</p>
          )}

          {result && (
            <div className="mt-4 p-4 rounded-lg bg-slate-50 border border-slate-200">
              {result.success ? (
                <>
                  <p className="text-2xl font-bold tracking-wide">
                    🚗 {result.plate_number}
                  </p>
                  <p className="text-sm text-slate-500 mt-1">
                    처리 시간 {result.processing_time}초 · 신뢰도 {result.confidence}
                  </p>
                </>
              ) : (
                <p className="text-slate-600">번호판을 찾지 못했습니다 ({result.processing_time}초)</p>
              )}
            </div>
          )}
        </section>

        <section className="bg-white rounded-2xl shadow p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <HistoryIcon className="w-5 h-5" />
            최근 인식 이력
          </h2>
          {history.length === 0 ? (
            <p className="text-sm text-slate-500">이력이 없습니다.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="text-left text-slate-500 border-b">
                  <tr>
                    <th className="py-2 pr-4">시각</th>
                    <th className="py-2 pr-4">번호판</th>
                    <th className="py-2 pr-4">원본파일</th>
                    <th className="py-2 pr-4">처리시간</th>
                    <th className="py-2 pr-4">결과</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((d) => (
                    <tr key={d.id} className="border-b last:border-0">
                      <td className="py-2 pr-4 text-slate-500">
                        {new Date(d.timestamp).toLocaleString('ko-KR')}
                      </td>
                      <td className="py-2 pr-4 font-semibold">{d.plate_number}</td>
                      <td className="py-2 pr-4 text-slate-500">
                        {d.original_filename ?? '-'}
                      </td>
                      <td className="py-2 pr-4 text-slate-500">
                        {d.processing_time ? `${d.processing_time.toFixed(2)}s` : '-'}
                      </td>
                      <td className="py-2 pr-4">
                        {d.result_image_url && (
                          <a
                            href={d.result_image_url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-slate-900 underline hover:text-slate-600"
                          >
                            보기
                          </a>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
