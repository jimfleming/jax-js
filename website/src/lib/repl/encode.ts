import { gunzipSync, gzipSync } from "fflate";
import { Base64 } from "js-base64";

/** Encode content as gzipped, URL-safe base64. */
export function encodeContent(code: string): string {
  const encoded = new TextEncoder().encode(code);
  const compressed = gzipSync(encoded, { mtime: 0 });
  const base64Content = Base64.fromUint8Array(compressed, true); // URL-safe base64
  return base64Content;
}

/** Decode content from base64 and unpack with gzip. */
export function decodeContent(encoded: string): string {
  const compressed = Base64.toUint8Array(encoded);
  const decompressed = gunzipSync(compressed);
  const decoded = new TextDecoder().decode(decompressed);
  return decoded;
}
