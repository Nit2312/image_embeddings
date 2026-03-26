import { Box, Stack, Typography } from '@mui/material'
import { useCallback, useMemo } from 'react'

export function FileDropzone(props: {
  label: string
  file: File | null
  onFileChange: (file: File | null) => void
  helperText?: string
}) {
  const { label, file, onFileChange, helperText } = props

  const previewUrl = useMemo(() => {
    if (!file) return null
    return URL.createObjectURL(file)
  }, [file])

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0] ?? null
      onFileChange(f)
    },
    [onFileChange]
  )

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      const f = e.dataTransfer.files?.[0] ?? null
      if (f) onFileChange(f)
    },
    [onFileChange]
  )

  return (
    <Stack spacing={1}>
      <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
        {label}
      </Typography>
      <Box
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        sx={{
          border: '1px dashed rgba(255,255,255,0.18)',
          borderRadius: 2,
          p: 2,
          background: 'rgba(255,255,255,0.03)',
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: '160px 1fr' },
          gap: 2,
          alignItems: 'center',
        }}
      >
        <Box
          sx={{
            width: 160,
            height: 160,
            borderRadius: 2,
            overflow: 'hidden',
            border: '1px solid rgba(255,255,255,0.08)',
            background: 'rgba(0,0,0,0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'text.secondary',
          }}
        >
          {previewUrl ? (
            <img
              src={previewUrl}
              alt={file?.name || 'preview'}
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
          ) : (
            <Typography variant="caption">Preview</Typography>
          )}
        </Box>

        <Stack spacing={0.75}>
          <Typography variant="body2" color="text.secondary">
            {helperText ?? 'Drop an image here or choose a file.'}
          </Typography>
          <input type="file" accept="image/*" onChange={onInputChange} />
          {file && (
            <Typography variant="caption" color="text.secondary">
              Selected: {file.name} ({Math.round(file.size / 1024)} KB)
            </Typography>
          )}
        </Stack>
      </Box>
    </Stack>
  )
}

