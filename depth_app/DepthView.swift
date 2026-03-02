// backend module for collection of training data


import SwiftUI
import ARKit
import UIKit

struct DepthView: UIViewRepresentable {
    @Binding var processState: String
    var onFrameCountUpdate: (UInt32) -> Void

    func makeUIView(context: Context) -> UIView {
        let container = UIView()

        // ARview setup
        let arView = ARSCNView(frame: .zero)
        arView.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(arView)

        NSLayoutConstraint.activate([
            arView.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            arView.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            arView.widthAnchor.constraint(equalTo: container.widthAnchor),
            arView.heightAnchor.constraint(equalTo: arView.widthAnchor, multiplier: 4.0 / 3.0)
        ])

        // configure lidar session
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = .sceneDepth
        arView.session.run(config)
        arView.delegate = context.coordinator

        return container
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // keep coordinator in sync with parent
        context.coordinator.currentState = processState

        // handle transitions between states
        switch processState {
        case "saving":
            if !context.coordinator.isSaving {
                context.coordinator.startSaving { _ in
                    DispatchQueue.main.async {
                        self.processState = "stopped"
                    }
                }
            }

        case "stopped":
            context.coordinator.clearFrames()
            context.coordinator.isSaving = false

        default:
            break
        }
    }


    func makeCoordinator() -> Coordinator {
        Coordinator(onFrameCountUpdate: onFrameCountUpdate)
    }

    // coordinator
    class Coordinator: NSObject, ARSCNViewDelegate {
        var onFrameCountUpdate: (UInt32) -> Void
        var currentState: String = "stopped"
        private let sharedContext = CIContext(options: nil)


        private var frameCount: UInt32 = 0 {
            didSet { onFrameCountUpdate(frameCount) }
        }

        init(onFrameCountUpdate: @escaping (UInt32) -> Void) {
            self.onFrameCountUpdate = onFrameCountUpdate
        }

        struct FramePair {
            let depth: [[Float32]]
            let image: UIImage   // 960x720, 8bit grayscale encoded
        }

        private(set) var recordedFrames: [FramePair] = []
        private var lastProcessedTime: TimeInterval = 0
        private var frameCounter = 0
        private var fpsTimerStart: TimeInterval = 0

        var isSaving = false

        // capture frames while recording
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard currentState == "record", !isSaving else { return }

            let minInterval: TimeInterval = 0.2
            if time - lastProcessedTime < minInterval { return }
            lastProcessedTime = time
            frameCount += 1

            guard
                let view = renderer as? ARSCNView,
                let frame = view.session.currentFrame,
                let depthMap = frame.sceneDepth?.depthMap
            else { return }

            // shallow copies of buffers, ARC-managed
            let capturedImageCopy: CVPixelBuffer = frame.capturedImage
            let depthMapCopy: CVPixelBuffer = frame.sceneDepth!.depthMap

            DispatchQueue.global(qos: .userInitiated).async {
                let ciImage = CIImage(cvPixelBuffer: capturedImageCopy)
                let context = self.sharedContext

                let targetWidth: CGFloat = 256 // output diminsion
                let targetHeight: CGFloat = 192
                let scaleX = targetWidth / ciImage.extent.width
                let scaleY = targetHeight / ciImage.extent.height
                let scaledCI = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

                guard let cgGray = context.createCGImage(
                    scaledCI,
                    from: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight),
                    format: .L8,
                    colorSpace: CGColorSpaceCreateDeviceGray()
                ) else { return }

                let grayImage = UIImage(cgImage: cgGray, scale: 1.0, orientation: .right)

                // depth extraction
                CVPixelBufferLockBaseAddress(depthMapCopy, .readOnly)
                let w = CVPixelBufferGetWidth(depthMapCopy)
                let h = CVPixelBufferGetHeight(depthMapCopy)
                guard let base = CVPixelBufferGetBaseAddress(depthMapCopy) else {
                    CVPixelBufferUnlockBaseAddress(depthMapCopy, .readOnly)
                    return
                }
                let ptr = base.assumingMemoryBound(to: Float32.self)
                var depthArray = Array(repeating: Array(repeating: Float32(0), count: w), count: h)
                for y in 0..<h {
                    for x in 0..<w {
                        depthArray[y][x] = ptr[y * w + x]
                    }
                }
                CVPixelBufferUnlockBaseAddress(depthMapCopy, .readOnly)

                // append atomically
                DispatchQueue.main.async {
                    self.recordedFrames.append(FramePair(depth: depthArray, image: grayImage))
                }
            }

        }


        // clear frames when stopped
        func clearFrames() {
            recordedFrames.removeAll()
            frameCount = 0
        }

        // save frames asynchronously
        func startSaving(completion: @escaping (Bool) -> Void) {
            guard !isSaving else { return }
            isSaving = true

            let framesToSave = recordedFrames
            recordedFrames.removeAll()

            guard !framesToSave.isEmpty else {
                isSaving = false
                completion(true)
                return
            }


            DispatchQueue.global(qos: .userInitiated).async {
                let tempDir = FileManager.default.temporaryDirectory
                    .appendingPathComponent("DepthData_\(UUID().uuidString)")
                do {
                    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
                } catch {
                    DispatchQueue.main.async {
                        self.isSaving = false
                        completion(false)
                    }
                    return
                }

                for (i, pair) in framesToSave.enumerated() {
                    // depth csv
                    let depthURL = tempDir.appendingPathComponent(String(format: "frame_%05d_depth.csv", i))
                    var csv = ""
                    for row in pair.depth {
                        csv += row.map { String(format: "%.4f", $0) }.joined(separator: ",") + "\n"
                    }
                    try? csv.write(to: depthURL, atomically: true, encoding: .utf8)

                    // grayscale png
                    let imageURL = tempDir.appendingPathComponent(String(format: "frame_%05d_gray.png", i))
                    if let pngData = pair.image.pngData() {
                        try? pngData.write(to: imageURL)
                    }
                }


                DispatchQueue.main.async {
                    let activityVC = UIActivityViewController(activityItems: [tempDir], applicationActivities: nil)
                    if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                       let rootVC = scene.windows.first?.rootViewController {
                        rootVC.present(activityVC, animated: true)
                    }
                    self.isSaving = false
                    self.frameCount = 0
                    completion(true)
                }
            }
        }
    }
}

