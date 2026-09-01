import fs from "fs";
import crypto from "crypto";

const input = JSON.parse(fs.readFileSync(process.argv[2]));
const hash = input.conformance_hash.slice(0, 12);

const svg = fs
  .readFileSync("template.svg", "utf8")
  .replace("{{GEOMETRY}}", input.geometry)
  .replace("{{KERNEL}}", input.kernel)
  .replace("{{HASH}}", hash);

fs.writeFileSync("badge.svg", svg);
