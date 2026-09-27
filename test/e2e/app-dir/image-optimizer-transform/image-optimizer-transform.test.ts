import { nextTestSetup } from 'e2e-utils'
import { readFile } from 'fs/promises'
import { join } from 'path'

// Finds the first SOF marker (0xc0 baseline, 0xc1 extended sequential, 0xc2
// progressive) and reads the frame width from the segment payload.
function findJpegSOF(buffer: Buffer): { width: number; progressive: boolean } {
  for (let i = 2; i + 9 < buffer.length; i++) {
    if (buffer[i] === 0xff) {
      const marker = buffer[i + 1]
      if (marker === 0xc0 || marker === 0xc1 || marker === 0xc2) {
        return {
          width: buffer.readUInt16BE(i + 7),
          progressive: marker === 0xc2,
        }
      }
    }
  }
  throw new Error('no JPEG SOF marker found')
}

describe('image-optimizer-transform', () => {
  const { next, isNextDev } = nextTestSetup({
    files: __dirname,
  })

  const getImage = (filename: string) =>
    readFile(join(__dirname, 'public', filename))

  function transform(img: string, query: Record<string, string> = {}) {
    return next.fetch(
      `/api/transform?${new URLSearchParams({ img, ...query })}`
    )
  }

  it('transforms a png buffer', async () => {
    const res = await transform('test.png', { mime: 'image/webp' })
    expect(res.status).toBe(200)
    expect(res.headers.get('content-type')).toBe('image/webp')
    expect(res.headers.get('x-max-age')).toBe('120')
    const body = Buffer.from(await res.arrayBuffer())
    expect(body.byteLength).toBeGreaterThan(0)
  })

  it.each([undefined, true, false])(
    'encodes JPEGs with imgOptMozjpeg=%s',
    async (mozjpeg) => {
      const query: Record<string, string> = { mime: 'image/jpeg' }
      if (mozjpeg !== undefined) {
        query.mozjpeg = String(mozjpeg)
      }
      const res = await transform('test.jpg', query)
      expect(res.status).toBe(200)
      expect(res.headers.get('content-type')).toBe('image/jpeg')
      const body = Buffer.from(await res.arrayBuffer())
      expect(findJpegSOF(body)).toEqual({
        width: 64,
        progressive: mozjpeg ?? true,
      })
    }
  )

  it('preserves the source format when no output format is requested', async () => {
    const res = await transform('test.png')
    expect(res.headers.get('content-type')).toBe('image/png')
  })

  it('transforms an avif source', async () => {
    const res = await transform('test.avif', { mime: 'image/webp' })
    expect(res.headers.get('content-type')).toBe('image/webp')
    const body = Buffer.from(await res.arrayBuffer())
    expect(body.equals(await getImage('test.avif'))).toBe(false)
  })

  it('downlevels an avif source when no output format is requested', async () => {
    const res = await transform('test.avif')
    expect(res.headers.get('content-type')).toBe('image/jpeg')
  })

  it('bypasses svg buffers', async () => {
    const res = await transform('test.svg', { mime: 'image/webp' })
    expect(res.headers.get('content-type')).toBe('image/svg+xml')
    const body = Buffer.from(await res.arrayBuffer())
    expect(body.equals(await getImage('test.svg'))).toBe(true)
  })

  it('bypasses animated buffers', async () => {
    const res = await transform('animated.webp', { mime: 'image/webp' })
    expect(res.headers.get('content-type')).toBe('image/webp')
    const body = Buffer.from(await res.arrayBuffer())
    expect(body.equals(await getImage('animated.webp'))).toBe(true)
  })

  it('rejects disallowed svg buffers', async () => {
    const res = await transform('test.svg', {
      mime: 'image/webp',
      allowSvg: 'false',
    })
    expect(res.status).toBe(400)
  })

  it('rejects unrecognized buffers', async () => {
    const res = await transform('bad-image.txt', { mime: 'image/webp' })
    expect(res.status).toBe(400)
  })

  // The dev placeholder flow is wired by the server's imageOptimizer wrapper
  // around the transform module, so these go through the real /_next/image
  // endpoint.
  ;(isNextDev ? describe : describe.skip)('dev blur placeholders', () => {
    const imageQuery = (url: string) =>
      `/_next/image?${new URLSearchParams({ url, w: '8', q: '70' })}`
    const acceptWebp = { headers: { accept: 'image/webp' } }

    it('generates blur placeholders in development', async () => {
      const res = await next.fetch(imageQuery('/test.png'), acceptWebp)
      expect(res.status).toBe(200)
      expect(res.headers.get('content-type')).toBe('image/svg+xml')
      expect(await res.text()).toContain('<svg')
    })

    it('does not generate blur placeholders for bypassed images', async () => {
      const res = await next.fetch(imageQuery('/test.svg'), acceptWebp)
      expect(res.status).toBe(200)
      expect(res.headers.get('content-type')).toBe('image/svg+xml')
      expect(await res.text()).toBe((await getImage('test.svg')).toString())
    })

    it('serves the placeholder from the optimizer cache on subsequent requests', async () => {
      const first = await next.fetch(imageQuery('/test.jpg'), acceptWebp)
      expect(first.headers.get('content-type')).toBe('image/svg+xml')
      expect(first.headers.get('x-nextjs-cache')).toBe('MISS')
      const second = await next.fetch(imageQuery('/test.jpg'), acceptWebp)
      expect(second.headers.get('content-type')).toBe('image/svg+xml')
      expect(second.headers.get('x-nextjs-cache')).toBe('HIT')
      expect(await second.text()).toBe(await first.text())
    })
  })
})
