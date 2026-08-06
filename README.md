# 双设备边缘视频分析系统

这是一个运行在真实设备上的视频流计算项目：树莓派 5（设备 A）采集视频，完成对比度增强、缩放、帧率控制和 JPEG 压缩；笔记本电脑（设备 B）通过 WebSocket 接收视频，运行 YOLO 目标检测、ByteTrack 人员跟踪、区域人数统计和人脸身份匹配，并在网页中展示画面、门禁身份和实时性能数据。

## 系统结构

```text
树莓派 5 / 设备 A                         笔记本 / 设备 B
摄像头 -> CLAHE -> 缩放 -> JPEG -> WebSocket -> 最新帧队列 -> YOLO/ByteTrack -> 人脸匹配 -> 网页监控
```

视频数据通过 Wi-Fi 这块无线网卡，承载为局域网里的 TCP/WebSocket 数据包，从树莓派发送到笔记本。项目不依赖互联网传输视频；Wi-Fi 提供本地无线链路，TCP/WebSocket 负责在这条链路上可靠传送视频帧。

设备 B 的接收队列只保留最新帧。当推理速度暂时低于发送速度时，系统主动丢弃旧帧，避免实时画面逐渐积累延迟。

## 环境要求

- 笔记本：Windows 10/11、Anaconda、Python 3.11
- 树莓派：64-bit Debian/Raspberry Pi OS、Python 3.11 或 3.13
- 两台设备网络互通，树莓派可以访问笔记本 TCP 端口 `8000`
- USB 摄像头或 Raspberry Pi CSI 摄像头

笔记本 AI 端固定使用 Python 3.11，以获得稳定的 PyTorch/Ultralytics 支持。树莓派设备端可以使用系统 Python 3.11 或 3.13；两端通过二进制协议通信，不要求 Python 小版本一致。树莓派使用轻量的 `venv`，CSI 摄像头场景通过 `--system-site-packages` 复用系统提供的 Picamera2。

## 1. 笔记本本地验证

在 Anaconda Prompt 或已经初始化 Conda 的 PowerShell 中进入工程目录：

```powershell
conda create -n edge_video python=3.11 pip -y
conda activate edge_video
python -m pip install --upgrade pip
python -m pip install -e ".[edge,dev]"
```

先启动不加载神经网络的传输测试服务：

```powershell
python -m edge_video.edge --detector mock
```

打开第二个 PowerShell 窗口，发送合成视频：

```powershell
python -m edge_video.device `
  --server ws://127.0.0.1:8000/ws/ingest/laptop-test `
  --source synthetic
```

浏览器打开 <http://127.0.0.1:8000>。看到移动色块和实时指标后，说明采集、预处理、传输、接收和网页显示链路正常。

## 2. 启动 AI 检测

停止 mock 服务，启动 YOLO。`--classes 0` 表示只检测 COCO 数据集中的人：

```powershell
python -m edge_video.edge `
  --detector yolo `
  --model yolo11n.pt `
  --classes 0 `
  --confidence 0.4 `
  --tracking `
  --roi 0.1,0.1,0.9,0.9
```

模型第一次运行会自动下载。要检测全部 80 类目标，传入空的类别参数：

```powershell
python -m edge_video.edge --detector yolo --classes ""
```

如果笔记本有可用的 NVIDIA GPU，可追加 `--device 0`；没有时默认使用 CPU。

ByteTrack 会为每个人分配持续的轨迹 ID。`--roi` 使用归一化坐标 `x1,y1,x2,y2` 定义统计区域，数值范围为 0 到 1。网页会显示区域内人数、活跃轨迹、累计进入/离开和最近事件；事件同时追加到 `artifacts/events.jsonl`。使用 `--no-tracking` 可以退回普通逐帧检测。

## 3. 树莓派部署

从远程仓库克隆代码后安装设备 A 所需依赖：

```bash
git clone <你的仓库地址>
cd <仓库目录>
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[device]"
```

检查 USB 摄像头：

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

如果没有 `v4l2-ctl`：

```bash
sudo apt update
sudo apt install v4l-utils
```

USB 摄像头通常使用 `--source 0`：

```bash
./.venv/bin/python -m edge_video.device \
  --server ws://<笔记本IP>:8000/ws/ingest/rpi5 \
  --source 0 \
  --camera-codec MJPG \
  --width 1280 \
  --height 720 \
  --max-width 960 \
  --fps 10 \
  --jpeg-quality 75
```

部分 Logitech UVC 摄像头的 MJPEG 帧会触发 `extraneous bytes before marker` 警告。画面通常仍可解码，但推荐改用无警告的 YUYV 配置：

```bash
./.venv/bin/python -m edge_video.device \
  --server ws://<笔记本IP>:8000/ws/ingest/rpi5 \
  --source 0 \
  --camera-codec YUYV \
  --width 960 \
  --height 544 \
  --max-width 960 \
  --fps 5 \
  --jpeg-quality 75
```

CSI 摄像头先验证系统识别：

```bash
rpicam-hello -t 5000
```

安装 Picamera2；上面创建的环境已经可以读取该系统包：

```bash
sudo apt update
sudo apt install python3-picamera2
```

随后使用：

```bash
./.venv/bin/python -m edge_video.device \
  --server ws://<笔记本IP>:8000/ws/ingest/rpi5 \
  --source picamera2
```

使用 `Ctrl+C` 停止任意一端程序。

## 4. 校园网连通性

在笔记本执行 `ipconfig`，找到当前 Wi-Fi 的 IPv4 地址。启动服务后，在树莓派测试：

```bash
curl http://<笔记本IP>:8000/health
```

预期响应类似：

```json
{"ok":true,"detector":"mock"}
```

如果无法访问：

1. 确认笔记本服务使用默认的 `--host 0.0.0.0`，而不是只监听 `127.0.0.1`。
2. 在 Windows 防火墙弹窗中允许 Python 接收连接，或为 TCP `8000` 创建入站规则。
3. 如果两端 IP 正确但相互不可达，校园网可能启用了终端隔离。改用同一个手机热点或网线直连。

### 证明视频通过 Wi-Fi 传输

在树莓派执行，其中 `172.20.10.3` 替换为笔记本当前 IP：

```bash
ip route get 172.20.10.3
```

当前设备的预期输出类似：

```text
172.20.10.3 dev wlan0 src 172.20.10.2 uid 1000
```

关键字段是 `dev wlan0`，表示发往笔记本的视频数据经过树莓派无线网卡；也就是视频帧虽然被封装成 TCP/WebSocket 数据包，但实际物理传输路径是 Wi-Fi。还可以补充展示：

```bash
ip -br address show wlan0
cat /sys/class/net/wlan0/operstate
```

`operstate` 输出 `up` 表示无线网卡正在工作。SSH 只是运行在这条 Wi-Fi 链路上的远程终端协议，并不是另一种物理连接；SSH 命令和 WebSocket 视频流都通过 `wlan0` 传输。

## 5. 共享网络上的访问令牌

校园网中建议为发送接口设置共享令牌。在两端分别设置相同值，再启动程序：

笔记本 PowerShell：

```powershell
$env:EDGE_STREAM_TOKEN="替换为足够长的随机字符串"
```

树莓派：

```bash
export EDGE_STREAM_TOKEN='替换为相同字符串'
```

令牌不会写入 Git。网页监控当前仍可被同网络设备访问，正式部署时还应结合防火墙限制来源地址。

## 6. 门禁人员登记与人脸识别

人脸检测和识别只在笔记本上运行，树莓派仍然只负责视频采集、预处理和发送，因此树莓派不需要安装人脸模型或新增 Python 依赖。笔记本使用 OpenCV YuNet 检测人脸，使用 SFace 提取特征并与已登记证件照进行余弦相似度匹配。

启动 `edge_video.edge` 时人脸识别默认开启。首次启动会把两个 OpenCV 官方 ONNX 模型下载到 `artifacts/models/`；中国大陆网络优先使用 Hugging Face 镜像，失败后回退到 GitHub。模型和人员数据都在 `.gitignore` 中，不会提交到远程仓库。临时关闭人脸功能可以追加：

```powershell
python -m edge_video.edge --no-face-recognition
```

网页登记流程：

1. 在浏览器打开 <http://127.0.0.1:8000>，找到右侧“门禁人员库”。
2. 输入姓名并选择一张正面、清晰、光线均匀的证件照，点击“登记人员”。
3. 对每个人重复登记。每张照片必须只包含一张人脸，大小不能超过 8 MB；姓名不允许重复。
4. 已登记人员进入摄像头画面后，视频框和“当前检测结果”会显示姓名；未登记或无法匹配的人显示 `stranger`。
5. 单击名单右侧的删除按钮并确认，即可删除该人员的照片和特征。

人员照片和特征默认持久化到 `artifacts/faces/`，重启服务后仍然有效。识别姓名会绑定到 ByteTrack 轨迹 ID，因此人员短暂转头或人脸被遮挡时，不会立即丢失已经确认的姓名；轨迹失效后缓存会自动清理。

默认匹配阈值为 OpenCV SFace 推荐的余弦相似度 `0.363`。演示环境中如果误识别较多，可以提高阈值，例如：

```powershell
python -m edge_video.edge --face-threshold 0.45
```

阈值越高越严格，误识别会减少，但光线、角度或摄像头清晰度较差时更容易显示 `stranger`。证件照与实时画面尽量保持相近的正脸角度。此功能适合课程演示，不应直接作为生产门锁的唯一认证因素；采集和保存人脸数据前应取得本人同意。

### 门禁功能演示建议

先登记两名人员并在网页名单中展示照片。让已登记人员进入画面，展示稳定的轨迹 ID 和姓名；再让一名未登记人员进入，展示 `stranger`。随后让多人同时出现在画面中，证明每个人员框可独立关联身份，并结合 ROI 的当前人数、累计进入和累计离开指标完成演示。

## 7. 测试

```powershell
conda activate edge_video
python -m pytest
python -m ruff check .
```

## 8. 树莓派进程控制与演示

在树莓派工程目录创建运行配置：

```bash
cd /home/brilliant/Documents/project/edge_video
cp .env.example .env
```

编辑 `.env`，至少设置正确的 `EDGE_SERVER_URL` 和与笔记本一致的 `EDGE_STREAM_TOKEN`。然后使用：

```bash
./scripts/device-control.sh start
./scripts/device-control.sh status
./scripts/device-control.sh stop
./scripts/device-control.sh restart
./scripts/device-control.sh logs
```

`start` 使用 `nohup` 将发送端放到后台，并把 PID 写入 `artifacts/device.pid`。进程不再依附 SSH 终端，因此关闭 SSH 或 VS Code Remote SSH 窗口后，视频仍会继续传输。可以用下面的命令证明进程没有终端：

```bash
ps -o pid,ppid,tty,stat,cmd -p "$(cat artifacts/device.pid)"
```

其中 `TTY` 为 `?` 表示进程不依赖 SSH 终端。

演示发送端停止和恢复：

```bash
./scripts/device-control.sh stop
# 网页状态变为“等待设备”
./scripts/device-control.sh start
# 发送端重新连接，网页恢复实时画面
```

这里的 `start` 是手动重新启动进程，WebSocket 连接会自动建立。真正的“断线自动重连”是指发送端仍在运行时，网络或笔记本服务暂时中断；发送端会持续重试，网络/服务恢复后无需重启树莓派进程即可重新连接。

演示断开树莓派 Wi-Fi 并自动重连：

```bash
sudo nmcli radio wifi off
# 等待 5 到 10 秒，观察网页画面停止更新或状态变为等待设备
sudo nmcli radio wifi on
# Wi-Fi 恢复后，不重启发送端进程，网页应自动恢复实时画面
```

如果当前是通过 SSH 连接树莓派，关闭 Wi-Fi 会同时断开 SSH。可以提前安排 10 秒后自动打开 Wi-Fi，再关闭无线：

```bash
(sleep 10; sudo nmcli radio wifi on) & sudo nmcli radio wifi off
```

## 9. 中国大陆网络下安装

普通 PyPI 依赖优先使用国内镜像，而不是依赖代理：

```powershell
python -m pip install -e ".[edge,dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

不建议永久修改全局 pip 配置，按命令指定镜像更容易排查问题。PyTorch 或 YOLO 模型文件下载缓慢时，可以使用稳定代理；终端需要继承系统代理，或者在当前 PowerShell 会话中临时设置：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:<代理端口>"
$env:HTTPS_PROXY="http://127.0.0.1:<代理端口>"
```

端口应替换为代理软件显示的本地 HTTP 代理端口。安装结束后可关闭当前终端，避免代理设置影响其他命令。

若树莓派的 `/etc/pip.conf` 配置了访问缓慢的 piwheels，可临时忽略全局配置：

```bash
PIP_CONFIG_FILE=/dev/null ./.venv/bin/python -m pip install setuptools wheel \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
PIP_CONFIG_FILE=/dev/null ./.venv/bin/python -m pip install --no-build-isolation \
  -e ".[device]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

发送端与服务端会在 WebSocket 建连时进行一次 NTP 式时钟同步。网页中的“时钟同步 RTT”是同步往返时间，“网络传输”和“端到端延迟”已经校正两台设备的系统时钟偏差。

## 演示与提交建议

最终实拍视频至少同时拍到树莓派/摄像头、被检测场景和笔记本实时网页。先展示端侧预处理参数，再展示人员进入画面后的检测框及性能指标；最后短暂断开网络并恢复，可以体现发送端自动重连能力。

提交仓库应保留清晰的提交历史。模型权重、虚拟环境和大体积录制视频已被 `.gitignore` 排除；视频可使用 GitHub/Gitee Release 或网盘链接提供。
