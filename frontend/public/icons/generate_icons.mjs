/**
 * Generates minimal PNG icons for the PWA using raw PNG binary encoding.
 * No external dependencies — runs with plain Node.js.
 * Usage: node generate_icons.mjs
 */

import { createHash } from "crypto";
import { writeFileSync } from "fs";
import { deflateSync } from "zlib";

function crc32(buf) {
  let crc = 0xffffffff;
  for (const byte of buf) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const typeBytes = Buffer.from(type, "ascii");
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const crcBuf = Buffer.concat([typeBytes, data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(crcBuf), 0);
  return Buffer.concat([len, typeBytes, data, crc]);
}

/**
 * Encode a flat RGBA pixel array into a PNG buffer.
 * @param {number} width
 * @param {number} height
 * @param {Uint8Array} rgba - width*height*4 bytes
 */
function encodePNG(width, height, rgba) {
  // IHDR
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 2;  // color type: RGB (we'll drop alpha for simplicity, use RGB)
  ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;

  // Build raw scanlines (filter byte 0 + RGB)
  const rowSize = 1 + width * 3;
  const raw = Buffer.alloc(height * rowSize);
  for (let y = 0; y < height; y++) {
    raw[y * rowSize] = 0; // filter type None
    for (let x = 0; x < width; x++) {
      const src = (y * width + x) * 4;
      const dst = y * rowSize + 1 + x * 3;
      raw[dst] = rgba[src];
      raw[dst + 1] = rgba[src + 1];
      raw[dst + 2] = rgba[src + 2];
    }
  }

  const compressed = deflateSync(raw, { level: 6 });

  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  return Buffer.concat([
    sig,
    chunk("IHDR", ihdr),
    chunk("IDAT", compressed),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function makeIcon(size, outputPath) {
  const pixels = new Uint8Array(size * size * 4);
  const cx = size / 2;
  const cy = size / 2;
  const outerR = size * 0.42;
  const innerR = size * 0.22;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const idx = (y * size + x) * 4;

      if (dist <= outerR && dist >= innerR) {
        // Blue ring
        pixels[idx] = 59;
        pixels[idx + 1] = 130;
        pixels[idx + 2] = 246;
        pixels[idx + 3] = 255;
      } else if (dist < innerR) {
        // Dark center
        pixels[idx] = 10;
        pixels[idx + 1] = 10;
        pixels[idx + 2] = 10;
        pixels[idx + 3] = 255;
      } else {
        // Black background
        pixels[idx] = 0;
        pixels[idx + 1] = 0;
        pixels[idx + 2] = 0;
        pixels[idx + 3] = 255;
      }
    }
  }

  const png = encodePNG(size, size, pixels);
  writeFileSync(outputPath, png);
  console.log(`Saved ${outputPath} (${size}x${size})`);
}

const dir = new URL(".", import.meta.url).pathname.replace(/^\/([A-Z]:)/, "$1");
makeIcon(192, `${dir}icon-192.png`);
makeIcon(512, `${dir}icon-512.png`);
makeIcon(180, `${dir}apple-touch-icon.png`);
