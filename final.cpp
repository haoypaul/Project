#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// 2D卷积
void conv2d(const std::vector<std::vector<std::vector<float>>>& input,
            std::vector<std::vector<std::vector<float>>>& output,
            const std::vector<std::vector<std::vector<std::vector<float>>>>& weight,
            const std::vector<float>& bias, int stride, int padding) {
    int in_channels = input.size();
    int out_channels = weight.size();
    int kernel_size = weight[0][0].size();
    int input_size = input[0][0].size();
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    output = std::vector<std::vector<std::vector<float>>>(out_channels,
              std::vector<std::vector<float>>(output_size, std::vector<float>(output_size, 0)));

    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels; ++i) {
            for (int y = 0; y < output_size; ++y) {
                for (int x = 0; x < output_size; ++x) {
                    float value = 0.0;
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int in_y = y * stride + ky - padding;
                            int in_x = x * stride + kx - padding;
                            if (in_y >= 0 && in_y < input_size && in_x >= 0 && in_x < input_size) {
                                value += input[i][in_y][in_x] * weight[o][i][ky][kx];
                            }
                        }
                    }
                    output[o][y][x] += value;
                }
            }
        }
        for (int y = 0; y < output_size; ++y) {
            for (int x = 0; x < output_size; ++x) {
                output[o][y][x] += bias[o];
            }
        }
    }
}

//ReLU
void relu(std::vector<std::vector<std::vector<float>>>& input) {
    for (auto& channel : input) {
        for (auto& row : channel) {
            for (auto& value : row) {
                if (value < 0) value = 0;
            }
        }
    }
}

// 全连接
void fully_connected(const std::vector<float>& input, std::vector<float>& output,
                     const std::vector<std::vector<float>>& weight, const std::vector<float>& bias) {
    int output_size = weight.size();
    int input_size = weight[0].size();

    for (int o = 0; o < output_size; ++o) {
        float value = 0.0;
        for (int i = 0; i < input_size; ++i) {
            value += input[i] * weight[o][i];
        }
        output[o] = value + bias[o];
    }
}

//加载权重
void load_weights(const std::string& filepath, std::vector<std::vector<std::vector<std::vector<float>>>>& conv_weights,
                  std::vector<float>& conv_bias, std::vector<std::vector<float>>& fc_weights, std::vector<float>& fc_bias) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open weights file!" << std::endl;
        return;
    }


    for (auto& o : conv_weights) {
        for (auto& i : o) {
            for (auto& ky : i) {
                file.read(reinterpret_cast<char*>(&ky[0]), ky.size() * sizeof(float));
            }
        }
    }
    file.read(reinterpret_cast<char*>(&conv_bias[0]), conv_bias.size() * sizeof(float));
    for (auto& o : fc_weights) {
        file.read(reinterpret_cast<char*>(&o[0]), o.size() * sizeof(float));
    }
    file.read(reinterpret_cast<char*>(&fc_bias[0]), fc_bias.size() * sizeof(float));

    file.close();
}

int main() {
    std::vector<std::vector<std::vector<float>>> input(3, std::vector<std::vector<float>>(32, std::vector<float>(32, 1.0f)));
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_weights(64, std::vector<std::vector<std::vector<float>>>(3,
        std::vector<std::vector<float>>(3, std::vector<float>(3))));
    std::vector<float> conv_bias(64);
    std::vector<std::vector<float>> fc_weights(10, std::vector<float>(512));
    std::vector<float> fc_bias(10);

    load_weights("checkpoint/resnet18_cifar10_int8.bin", conv_weights, conv_bias, fc_weights, fc_bias);
    std::vector<std::vector<std::vector<float>>> conv_output;
    conv2d(input, conv_output, conv_weights, conv_bias, 1, 1);

    // 激活函数
    relu(conv_output);


    std::vector<float> flattened;
    for (const auto& channel : conv_output) {
        for (const auto& row : channel) {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }
    }

    std::vector<float> fc_output(10);
    fully_connected(flattened, fc_output, fc_weights, fc_bias);

    // 输出结果
    std::cout << "Final output: ";
    for (const auto& value : fc_output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
