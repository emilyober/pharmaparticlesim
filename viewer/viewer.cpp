#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct DisplayParticle {
    Vector3 position;
    float radius;
    int group;
};

struct Frame {
    double time;
    std::vector<DisplayParticle> particles;
};

std::vector<Frame> loadFrames(const std::string& filename){
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open " + filename);
    }
    std::string line;
    std::getline(file,line);
    if(!line.empty() && line.back() == 'r'){
        line.pop_back();
    }
    const std::string expectedHeader = "step,time,id,group,x,y,z,vx,vy,vz,"
        "omega_x,omega_y,omega_z,radius";
    if (line != expectedHeader){
        throw std::runtime_error(
            "Unexpected CSV header. Expected the group column."
        );
    }
    std::vector<Frame> frames;
    std::string previousStep;
    while(std::getline(file,line)){
        if (line.empty() || line == "\r"){
            continue;
        }
        std::stringstream stream(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(stream,field,',')){
            fields.push_back(field);
        }
        if(fields.size() != 14){
            throw std::runtime_error("Invalid CSV row.");
        }
        const double time = std::stod(fields[1]);
        if (frames.empty() || fields[0] != previousStep){
            if(!frames.empty()&&time <= frames.back().time){
                throw std::runtime_error(
                    "Frame times must increase."
                );
            }
            frames.push_back(Frame{time,{}});
            previousStep = fields[0];
        }
        const float x = std::stof(fields[4]) * 1000.0f;
        const float y = std::stof(fields[5]) * 1000.0f;
        const float z = std::stof(fields[6]) * 1000.0f;
        const float radius = std::stof(fields[13]) * 1000.0f;
        if (!std::isfinite(x) ||
            !std::isfinite(y) ||
            !std::isfinite(z) ||
            !std::isfinite(radius) ||
            radius <= 0.0f ||
            !std::isfinite(time)) {
            throw std::runtime_error("Invalid particle data.");
        }
        frames.back().particles.push_back({Vector3{x,z,-y},radius,std::stoi(fields[3])});
    }
    if (frames.empty()) {
        throw std::runtime_error("CSV contains no particles.");
    }
    return frames;
};

void drawDrum(float radius, float length) {
    constexpr int segments = 64;
    constexpr float pi = 3.141592653589793234f;
    const float halfLength = length *0.5f;
    for (int i = 0; i < segments; ++i){
        const float a = 2.0f * pi * i/segments;
        const float b = 2.0f * pi * (i+1)/segments;
        const Vector3 frontA{radius*std::cos(a),radius*std::sin(a),-halfLength};
        const Vector3 frontB{radius*std::cos(b),radius*std::sin(a),-halfLength};
        const Vector3 backA{frontA.x, frontA.y, halfLength};
        const Vector3 backB{frontB.x, frontB.y, halfLength};
        DrawLine3D(frontA, frontB, GRAY);
        DrawLine3D(backA, backB, GRAY);

        if (i % 8 == 0) {
            DrawLine3D(frontA, backA, LIGHTGRAY);
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        const std::string filename =
            argc > 1 ? argv[1] : "drum_mixing.csv";

        const std::vector<Frame> frames =
            loadFrames(filename);

        InitWindow(1100, 800, "DEM drum viewer");
        SetTargetFPS(60);

        Camera3D camera{};
        camera.target = Vector3{0.0f, 0.0f, 0.0f};
        camera.up = Vector3{0.0f, 1.0f, 0.0f};
        camera.fovy = 45.0f;
        camera.projection = CAMERA_PERSPECTIVE;

        float yaw = 0.5f;
        float pitch = 0.3f;
        float distance = 80.0f;

        bool paused = false;
        std::size_t frameIndex = 0;
        double playTime = frames.front().time;

        //0.05 simulated seconds per real second.
        //0.1-second run plays in about 2 seconds.
        constexpr double playbackSpeed = 0.05;

        while (!WindowShouldClose()) {
            if (IsKeyPressed(KEY_SPACE)) {
                paused = !paused;
            }

            if (IsKeyPressed(KEY_R)) {
                playTime = frames.front().time;
                frameIndex = 0;
            }

            if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
                const Vector2 delta = GetMouseDelta();

                yaw -= delta.x * 0.005f;
                pitch += delta.y * 0.005f;
                pitch = std::clamp(pitch, -1.4f, 1.4f);
            }

            distance *=
                std::exp(-GetMouseWheelMove() * 0.1f);

            distance = std::clamp(distance, 30.0f, 200.0f);

            camera.position = Vector3{
                distance * std::cos(pitch) * std::sin(yaw),
                distance * std::sin(pitch),
                distance * std::cos(pitch) * std::cos(yaw)
            };

            if (!paused && frames.size() > 1) {
                playTime += GetFrameTime() * playbackSpeed;

                if (playTime > frames.back().time) {
                    playTime = frames.front().time;
                    frameIndex = 0;
                }

                while (frameIndex + 1 < frames.size() &&
                       frames[frameIndex + 1].time <= playTime) {
                    ++frameIndex;
                }
            }

            const Frame& frame = frames[frameIndex];

            BeginDrawing();
            ClearBackground(RAYWHITE);

            BeginMode3D(camera);

            //display units are millimetres.
            //match the 20 mm radius, 20 mm long drum.
            drawDrum(20.0f, 20.0f);

            for (const DisplayParticle& particle :
                 frame.particles) {
                const Color color =
                    particle.group == 0 ? BLUE : ORANGE;

                DrawSphereEx(
                    particle.position,
                    particle.radius,
                    8,
                    12,
                    color
                );
            }

            EndMode3D();

            DrawText(
                TextFormat(
                    "Simulated time: %.5f s   Particles: %d",
                    frame.time,
                    static_cast<int>(frame.particles.size())
                ),
                20, 20, 20, DARKGRAY
            );

            DrawText(
                "Space: pause | R: restart | "
                "Right-drag: orbit | Scroll: zoom",
                20, 50, 18, DARKGRAY
            );

            DrawText(
                paused ? "PAUSED" : "Slow-motion playback",
                20, 80, 18, DARKGRAY
            );

            EndDrawing();
        }

        CloseWindow();
    } catch (const std::exception& error) {
        std::cerr << "Viewer error: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}
