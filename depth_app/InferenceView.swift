// backend module for running inference with the trained coreML model.


import SwiftUI
import ARKit
import UIKit
import CoreML

struct InferenceView: UIViewRepresentable {

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> UIView {
        let container = UIView()


        let arView = ARSCNView(frame: .zero)
        arView.isHidden = true    
        arView.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(arView)
        context.coordinator.arView = arView

        NSLayoutConstraint.activate([
            arView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            arView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            arView.topAnchor.constraint(equalTo: container.topAnchor),
            arView.bottomAnchor.constraint(equalTo: container.bottomAnchor)
        ])

        let outView = UIImageView()
        outView.translatesAutoresizingMaskIntoConstraints = false
        outView.contentMode = .scaleAspectFit
        outView.backgroundColor = .black
        container.addSubview(outView)
        context.coordinator.outputView = outView

        NSLayoutConstraint.activate([
            outView.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            outView.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            outView.widthAnchor.constraint(equalToConstant: 192),
            outView.heightAnchor.constraint(equalToConstant: 256)
        ])

        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = []            // no LiDAR
        arView.session.run(config)
        arView.delegate = context.coordinator

        return container
    }

    func updateUIView(_ uiView: UIView, context: Context) {}


    class Coordinator: NSObject, ARSCNViewDelegate {

        weak var arView: ARSCNView?            // hidden camera capture
        weak var outputView: UIImageView?      // visible depth output

        private let sharedContext = CIContext(options: nil)
        private var lastTime: TimeInterval = 0

        private let model: DepthModel = {
            do { return try DepthModel(configuration: MLModelConfiguration()) }
            catch { fatalError("Failed to load ML model") }
        }()

        // convert CGImage to MLMultiArray
        static func imageToMultiArray(_ cgImage: CGImage) -> MLMultiArray? {
            let width = cgImage.width
            let height = cgImage.height

            guard let arr = try? MLMultiArray(
                shape: [1, 1, NSNumber(value: height), NSNumber(value: width)],
                dataType: .float32
            ) else { return nil }

            guard let data = cgImage.dataProvider?.data,
                  let ptr = CFDataGetBytePtr(data)
            else { return nil }

            let bytesPerRow = cgImage.bytesPerRow
            let out = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))

            for y in 0..<height {
                for x in 0..<width {
                    out[y * width + x] = Float32(ptr[y * bytesPerRow + x]) / 255.0
                }
            }
            return arr
        }

        // convert output MLMultiArray to grayscale UIImage
        static func depthToCGImage(_ arr: MLMultiArray) -> CGImage? {
            let height = arr.shape[2].intValue
            let width  = arr.shape[3].intValue

            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height)
            let src = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))

            for i in 0..<(width * height) {
                let v = max(0, min(1, src[i]))
                buffer[i] = UInt8(v * 255)
            }

            let cs = CGColorSpaceCreateDeviceGray()
            return CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 8,
                bytesPerRow: width,
                space: cs,
                bitmapInfo: [],
                provider: CGDataProvider(
                    dataInfo: nil,
                    data: buffer,
                    size: width * height,
                    releaseData: { _, p, _ in p.deallocate() }
                )!,
                decode: nil, shouldInterpolate: false,
                intent: .defaultIntent
            )
        }


        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {

            guard let frame = arView?.session.currentFrame else { return }
            let captured = frame.capturedImage

            DispatchQueue.global(qos: .userInitiated).async {

                let ci = CIImage(cvPixelBuffer: captured)

                // rotate camera to portrait orientation
                let portraitCI = ci.oriented(.right)

                let targetWidth: CGFloat = 192
                let targetHeight: CGFloat = 256

                let sx = targetWidth / portraitCI.extent.width
                let sy = targetHeight / portraitCI.extent.height
                let scaled = portraitCI.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

                guard let cgGray = self.sharedContext.createCGImage(
                    scaled,
                    from: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight),
                    format: CIFormat.L8,
                    colorSpace: CGColorSpaceCreateDeviceGray()
                ) else { return }

                guard let multi = Self.imageToMultiArray(cgGray) else { return }
                guard let out = try? self.model.prediction(x: multi) else { return }
                guard let cgOut = Self.depthToCGImage(out.var_341) else { return }

                DispatchQueue.main.async {
                    self.outputView?.image = UIImage(cgImage: cgOut)
                }
            }


        }
    }
}
