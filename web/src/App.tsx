import { useMemo, useState } from 'react'
import {
  AppBar,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Container,
  Divider,
  FormControlLabel,
  LinearProgress,
  Stack,
  Switch,
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material'
import SearchIcon from '@mui/icons-material/Search'
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import { addVector, deleteVector, searchByImage } from './api/vectorizeWorker'
import type { WorkerMatch } from './api/vectorizeWorker'
import { MatchGrid } from './components/MatchGrid'
import { FileDropzone } from './components/FileDropzone'

type TabKey = 'search' | 'ingest' | 'admin'

function App() {
  const [tab, setTab] = useState<TabKey>('search')

  // Search
  const [queryFile, setQueryFile] = useState<File | null>(null)
  const [topK, setTopK] = useState<number>(20)
  const [productIdFilter, setProductIdFilter] = useState<string>('')
  const [onlyProducts, setOnlyProducts] = useState<boolean>(true)
  const [searching, setSearching] = useState(false)
  const [matches, setMatches] = useState<WorkerMatch[]>([])
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchMs, setSearchMs] = useState<number | null>(null)

  // Ingest
  const [ingestFile, setIngestFile] = useState<File | null>(null)
  const [ingestId, setIngestId] = useState<string>('')
  const [ingestProductId, setIngestProductId] = useState<string>('')
  const [ingesting, setIngesting] = useState(false)
  const [ingestResult, setIngestResult] = useState<string | null>(null)

  // Admin
  const [deleteId, setDeleteId] = useState<string>('')
  const [deleting, setDeleting] = useState(false)
  const [deleteResult, setDeleteResult] = useState<string | null>(null)

  const filter = useMemo(() => {
    const f: Record<string, any> = {}
    if (onlyProducts) {
      // only apply if metadata.product_id exists (heuristic via $ne null)
      f.product_id = { $ne: null }
    }
    if (productIdFilter.trim()) {
      f.product_id = { $eq: productIdFilter.trim() }
    }
    return Object.keys(f).length ? f : undefined
  }, [onlyProducts, productIdFilter])

  async function onSearch() {
    setSearchError(null)
    setMatches([])
    setSearchMs(null)
    if (!queryFile) {
      setSearchError('Pick an image to search.')
      return
    }

    setSearching(true)
    try {
      const start = performance.now()
      const res = await searchByImage({ file: queryFile, topK, filter })
      setMatches(res.matches)
      setSearchMs(Math.round(performance.now() - start))
    } catch (e: any) {
      setSearchError(e?.message || 'Search failed')
    } finally {
      setSearching(false)
    }
  }

  async function onIngest() {
    setIngestResult(null)
    if (!ingestFile) {
      setIngestResult('Pick an image to add.')
      return
    }

    setIngesting(true)
    try {
      const metadata: Record<string, any> = {
        source: 'ui',
      }
      if (ingestProductId.trim()) metadata.product_id = ingestProductId.trim()

      const res = await addVector({
        file: ingestFile,
        id: ingestId.trim() ? ingestId.trim() : undefined,
        metadata,
      })
      setIngestResult(`Added. id=${res.id} mutationId=${res.mutationId}`)
      setIngestFile(null)
      setIngestId('')
    } catch (e: any) {
      setIngestResult(e?.message || 'Add failed')
    } finally {
      setIngesting(false)
    }
  }

  async function onDelete() {
    setDeleteResult(null)
    if (!deleteId.trim()) {
      setDeleteResult('Enter a vector id to delete.')
      return
    }
    setDeleting(true)
    try {
      const res = await deleteVector({ id: deleteId.trim() })
      setDeleteResult(`Delete requested. id=${res.id} mutationId=${res.mutationId}`)
    } catch (e: any) {
      setDeleteResult(e?.message || 'Delete failed')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <Box sx={{ minHeight: '100vh' }}>
      <AppBar position="sticky" elevation={0} sx={{ backdropFilter: 'blur(12px)', background: 'rgba(15,22,41,0.7)' }}>
        <Container maxWidth="lg">
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ py: 1 }}>
            <Stack spacing={0.2}>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                Retail Visual AI
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Visual search • Duplicate detection • Catalog curation (Vectorize)
              </Typography>
            </Stack>
            <Chip
              size="small"
              label={`Worker: ${import.meta.env.VITE_WORKER_URL ?? '(set VITE_WORKER_URL)'}`}
              variant="outlined"
              sx={{ maxWidth: 420 }}
            />
          </Stack>
        </Container>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Card sx={{ border: '1px solid rgba(255,255,255,0.08)' }}>
          <CardHeader
            title="Catalog Vector Tools"
            subheader="Upload product images into Vectorize and perform similarity search to power retail workflows."
          />
          <CardContent>
            <Tabs
              value={tab}
              onChange={(_, v) => setTab(v)}
              textColor="primary"
              indicatorColor="primary"
              sx={{ mb: 2 }}
            >
              <Tab value="search" label="Visual Search" icon={<SearchIcon />} iconPosition="start" />
              <Tab value="ingest" label="Add to Catalog" icon={<AddPhotoAlternateIcon />} iconPosition="start" />
              <Tab value="admin" label="Admin" icon={<DeleteOutlineIcon />} iconPosition="start" />
            </Tabs>

            {tab === 'search' && (
              <Stack spacing={2}>
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                  <Box sx={{ flex: 1 }}>
                    <FileDropzone
                      label="Query image"
                      file={queryFile}
                      onFileChange={setQueryFile}
                      helperText="Drop an image or click to select. We’ll return similar catalog items."
                    />
                  </Box>
                  <Stack spacing={2} sx={{ width: { xs: '100%', md: 360 } }}>
                    <TextField
                      label="Top K"
                      type="number"
                      value={topK}
                      onChange={(e) => setTopK(Math.max(1, Math.min(100, Number(e.target.value || 20))))}
                      inputProps={{ min: 1, max: 100 }}
                    />
                    <TextField
                      label="Filter by product_id (optional)"
                      placeholder="e.g. 12345"
                      value={productIdFilter}
                      onChange={(e) => setProductIdFilter(e.target.value)}
                    />
                    <FormControlLabel
                      control={<Switch checked={onlyProducts} onChange={(e) => setOnlyProducts(e.target.checked)} />}
                      label="Prefer items with product_id metadata"
                    />
                    <Button variant="contained" onClick={onSearch} disabled={searching} startIcon={<SearchIcon />}>
                      Search
                    </Button>
                  </Stack>
                </Stack>

                {searching && <LinearProgress />}
                {searchError && (
                  <Typography color="error" variant="body2">
                    {searchError}
                  </Typography>
                )}
                {searchMs !== null && !searchError && (
                  <Typography variant="caption" color="text.secondary">
                    Search took {searchMs} ms
                  </Typography>
                )}

                <Divider />
                <MatchGrid matches={matches} />
              </Stack>
            )}

            {tab === 'ingest' && (
              <Stack spacing={2}>
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                  <Box sx={{ flex: 1 }}>
                    <FileDropzone
                      label="Catalog image"
                      file={ingestFile}
                      onFileChange={setIngestFile}
                      helperText="Adds a vector with metadata so it becomes searchable."
                    />
                  </Box>
                  <Stack spacing={2} sx={{ width: { xs: '100%', md: 360 } }}>
                    <TextField
                      label="Vector id (optional)"
                      placeholder="leave empty to auto-generate"
                      value={ingestId}
                      onChange={(e) => setIngestId(e.target.value)}
                    />
                    <TextField
                      label="product_id (recommended)"
                      placeholder="e.g. SKU / product id"
                      value={ingestProductId}
                      onChange={(e) => setIngestProductId(e.target.value)}
                    />
                    <Button variant="contained" onClick={onIngest} disabled={ingesting} startIcon={<AddPhotoAlternateIcon />}>
                      Add Vector
                    </Button>
                  </Stack>
                </Stack>

                {ingesting && <LinearProgress />}
                {ingestResult && (
                  <Typography variant="body2" color={ingestResult.toLowerCase().includes('added') ? 'text.primary' : 'error'}>
                    {ingestResult}
                  </Typography>
                )}
              </Stack>
            )}

            {tab === 'admin' && (
              <Stack spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  For catalog operations, prefer stable ids (e.g. hash of R2 key) so re-uploads overwrite cleanly.
                </Typography>
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems="flex-start">
                  <TextField
                    label="Delete vector id"
                    placeholder="e.g. 32-char sha256 prefix"
                    value={deleteId}
                    onChange={(e) => setDeleteId(e.target.value)}
                    sx={{ flex: 1 }}
                  />
                  <Button variant="outlined" onClick={onDelete} disabled={deleting} startIcon={<DeleteOutlineIcon />}>
                    Delete
                  </Button>
                </Stack>
                {deleting && <LinearProgress />}
                {deleteResult && (
                  <Typography variant="body2" color={deleteResult.toLowerCase().includes('requested') ? 'text.primary' : 'error'}>
                    {deleteResult}
                  </Typography>
                )}
              </Stack>
            )}
          </CardContent>
        </Card>
      </Container>
    </Box>
  )
}

export default App
