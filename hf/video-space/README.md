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

Pick a football match video and the page works out which team had the ball, five times a second, and reads the pitch markings to place every player in shot on a plan of the pitch.

What you get, with numbers that follow the video as it plays:

- possession, a momentum chart, how long the ball was really in play, balls lost and won back within five seconds, pressing, and which side had more players around the ball
- the pitch from above: where each team had the ball, both teams' width and the height of their back lines
- passes and passing moves, with a pass map for each team
- shots and expected goals (xG). The page suggests shots for you to confirm or remove, you can add the ones it missed, and the xG adds up over time
- moments picked out of the match, each ready to save as a video clip, a short written summary, and the whole stretch as a video with the marks on it

You can save a picture of the result to share, and the data as CSV and JSON. The page is in Arabic and English. It all runs in your browser; the video is never uploaded.

It is fastest in a browser with WebGPU, such as a recent Chrome or Edge on a laptop or desktop. Without WebGPU it runs on the processor, several times slower, so start with a few minutes of video.

The pitch view needs wide shots from the main camera with the pitch markings in view. The xG model is a logistic regression on shot distance and angle, fitted on 12,313 shots from StatsBomb open data (https://github.com/statsbomb/open-data).

Code and method: https://github.com/amr7772/possession-from-tracking (the page itself is hf/video-space/index.html there)

The same project on tracking data: https://huggingface.co/spaces/amreleraqi09/possession-from-tracking

The detector is YOLO26s by Ultralytics, licensed under AGPL-3.0, run with ONNX Runtime Web.
