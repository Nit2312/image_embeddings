import { Box, Card, CardContent, Chip, Stack, Typography } from '@mui/material'
import type { WorkerMatch } from '../api/vectorizeWorker'

function imageUrlFromMetadata(metadata?: Record<string, any>): string | null {
  const base = (import.meta.env.VITE_IMAGE_BASE_URL as string | undefined) ?? ''
  if (!base) return null
  const key = metadata?.relative_path || metadata?.filename
  if (!key) return null
  return `${base.replace(/\/+$/, '')}/${String(key).replace(/^\/+/, '')}`
}

export function MatchGrid(props: { matches: WorkerMatch[] }) {
  const { matches } = props

  if (!matches.length) {
    return (
      <Box sx={{ py: 4, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          No matches yet. Run a search to see similar catalog items.
        </Typography>
      </Box>
    )
  }

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: {
          xs: '1fr',
          sm: 'repeat(2, minmax(0, 1fr))',
          md: 'repeat(3, minmax(0, 1fr))',
          lg: 'repeat(4, minmax(0, 1fr))',
        },
        gap: 2,
      }}
    >
      {matches.map((m, idx) => {
        const img = imageUrlFromMetadata(m.metadata)
        const filename = m.metadata?.filename || m.metadata?.local_filename || m.metadata?.relative_path || m.id
        const productId = m.metadata?.product_id
        const source = m.metadata?.source

        return (
          <Card
            key={`${m.id}-${idx}`}
            sx={{
              height: '100%',
              border: '1px solid rgba(255,255,255,0.08)',
              background: 'rgba(255,255,255,0.03)',
            }}
          >
            <Box
              sx={{
                height: 160,
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                background: 'rgba(0,0,0,0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
              }}
            >
              {img ? (
                <img
                  src={img}
                  alt={String(filename)}
                  loading="lazy"
                  decoding="async"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    width: 'auto',
                    height: 'auto',
                    objectFit: 'contain',
                  }}
                />
              ) : (
                <Typography variant="caption" color="text.secondary">
                  Set VITE_IMAGE_BASE_URL to show images
                </Typography>
              )}
            </Box>
            <CardContent>
              <Stack spacing={1}>
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                  <Chip size="small" label={`#${idx + 1}`} variant="outlined" />
                  <Chip size="small" color="primary" label={`${(m.score * 100).toFixed(2)}%`} />
                  {productId && <Chip size="small" color="secondary" label={`product_id: ${productId}`} />}
                  {source && <Chip size="small" label={String(source)} variant="outlined" />}
                </Stack>

                <Typography variant="subtitle2" sx={{ fontWeight: 700 }} noWrap title={String(filename)}>
                  {String(filename)}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ wordBreak: 'break-all' }}>
                  id: {m.id}
                </Typography>
              </Stack>
            </CardContent>
          </Card>
        )
      })}
    </Box>
  )
}

