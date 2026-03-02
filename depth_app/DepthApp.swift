// This code collects pairs of image data and lidar-based depth data at 5fps and saves them to the mobile device filesystem.
// The collected data is used to train depth prediction models off-device.
//

import SwiftUI
import SwiftData

@main
struct DepthApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
