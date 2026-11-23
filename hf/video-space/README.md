---
title: Who has the ball? From video
emoji: ⚽
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: agpl-3.0
short_description: Football possession from a match video, in your browser
---

Pick a football match video and the page works out which team had the ball, five times a second: the split, a timeline of every spell, and the video with each player's team and the ball marked. It all runs in your browser. The video is never uploaded.

It is fastest in a browser with WebGPU, such as a recent Chrome or Edge on a laptop or desktop. Without WebGPU it runs on the processor, several times slower, so start with a few minutes of video.

Code and method: https://github.com/amr7772/possession-from-tracking (the page itself is hf/video-space/index.html there)

The same project on tracking data: https://huggingface.co/spaces/amreleraqi09/possession-from-tracking

The detector is YOLO26s by Ultralytics, licensed under AGPL-3.0, run with ONNX Runtime Web.
