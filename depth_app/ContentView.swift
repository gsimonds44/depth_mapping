// GUI

import SwiftUI
import ARKit
import UIKit

struct ContentView: View {
    @State private var processState = "stopped"
    @State private var frameCount: UInt32 = 0
    @State private var inferenceMode = false

    
    var body: some View {
        ZStack {
            if inferenceMode {
                InferenceView()
            } else {
                DepthView(processState: $processState) { newCount in
                    frameCount = newCount
                }
            }



            VStack {
                // --- TOP BUTTON ---
                Button(action: { inferenceMode.toggle() }) {
                    Text(inferenceMode ? "switch to recording" : "switch to inference")
                        .padding()
                        .background(Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding(.top, 40)

                Spacer()

                // --- BOTTOM BUTTON (only if NOT in inference mode) ---
                if !inferenceMode {
                    Button(action: handleButtonPress) {
                        Text(buttonLabel)
                            .padding()
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .disabled(processState == "saving")
                    .padding(.bottom, 40)
                }
            }
        }
    }


    // derived button text
    private var buttonLabel: String {
        switch processState {
        case "stopped": return "start recording"
        case "record":  return "save \(frameCount) frames"
        case "saving":  return "saving..."
        default:        return "start recording"
        }
    }

    // button action logic
    private func handleButtonPress() {
        switch processState {
        case "stopped":
            processState = "record"
        case "record":
            processState = "saving"
        default:
            break
        }
    }
}

