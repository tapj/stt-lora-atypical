# FFmpeg binaries

Place a static FFmpeg build in this directory so the application can transcode browser uploads. For example:

```bash
curl -L http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg-release-amd64-static.tar.xz
dir=$(tar -tf ffmpeg-release-amd64-static.tar.xz | head -n 1 | cut -d/ -f1)
tar -xf ffmpeg-release-amd64-static.tar.xz -C bin
cd bin && ln -sf \"$dir/ffmpeg\" ffmpeg && ln -sf \"$dir/ffprobe\" ffprobe
rm ffmpeg-release-amd64-static.tar.xz
```

After installing, verify the binaries are on your path:

```bash
PATH=./bin:${PATH} ffmpeg -version
```
