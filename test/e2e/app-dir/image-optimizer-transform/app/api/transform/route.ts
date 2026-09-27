import { readFile } from 'fs/promises'
import { join } from 'path'
import {
  ImageError,
  imageOptimizerTransform,
  type ImageOptimizerTransformConfig,
} from 'next/image-optimizer-transform'

export async function GET(request: Request) {
  const url = new URL(request.url)
  const img = url.searchParams.get('img')
  if (img === null) {
    return new Response('missing img param', { status: 400 })
  }

  const mozjpeg = url.searchParams.get('mozjpeg')
  const config: ImageOptimizerTransformConfig = {
    images: {
      dangerouslyAllowSVG: url.searchParams.get('allowSvg') !== 'false',
      minimumCacheTTL: 60,
    },
    experimental: {
      imgOptConcurrency: 1,
      imgOptOperationCache: false,
      imgOptMaxInputPixels: 67_108_864,
      imgOptSequentialRead: true,
      imgOptTimeoutInSeconds: 6,
      ...(mozjpeg === null ? {} : { imgOptMozjpeg: mozjpeg === 'true' }),
    },
  }

  const buffer = await readFile(join(process.cwd(), 'public', img))

  try {
    const result = await imageOptimizerTransform(
      {
        buffer,
        contentType: undefined,
        cacheControl: 'public, max-age=120',
        etag: 'source-etag',
      },
      {
        href: `/${img}`,
        width: Number(url.searchParams.get('w') ?? 64),
        quality: Number(url.searchParams.get('q') ?? 75),
        mimeType: url.searchParams.get('mime') ?? '',
      },
      config
    )

    return new Response(new Uint8Array(result.buffer), {
      headers: {
        'content-type': result.contentType,
        'x-max-age': String(result.maxAge),
        'x-upstream-etag': result.upstreamEtag,
      },
    })
  } catch (err) {
    if (err instanceof ImageError) {
      return new Response(err.message, { status: err.statusCode })
    }
    throw err
  }
}
