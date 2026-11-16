"""TTFAB and RTF of one tts/server.py config, as the orchestrator calls it (POST /v1/tts, streamed PCM16).

    python v2_ttfab.py <label>   # in the orchestrator image, sharing the tts container's network
The 15 unseen sentences (product text, split_final.json), one at a time, after one warmup pass over all.
"""
import asyncio, json, statistics, sys, time

import aiohttp

URL = "http://localhost:8000/v1/tts"


async def one(session, text):
    t0 = time.monotonic()
    first, n = None, 0
    async with session.post(URL, json={"text": text}) as r:
        r.raise_for_status()
        rate = int(r.headers["X-Sample-Rate"])
        async for chunk in r.content.iter_any():
            first = first or time.monotonic() - t0
            n += len(chunk)
    return first, (time.monotonic() - t0) / (n / 2 / rate)


async def main():
    texts = [s["final"] for s in json.load(open("/d/split_final.json"))["sentences"] if s["sid"][0] == "u"]
    async with aiohttp.ClientSession() as session:
        for t in texts:  # torch.compile settles per shape; the live server is warm too after a few visitors
            await one(session, t)
        res = [await one(session, t) for t in texts]
    ttfab = sorted(r[0] * 1000 for r in res)
    print(json.dumps({"config": sys.argv[1], "ttfab_med_ms": round(statistics.median(ttfab)),
                      "ttfab_max_ms": round(ttfab[-1]), "rtf_med": round(statistics.median(r[1] for r in res), 3)}))

asyncio.run(main())
