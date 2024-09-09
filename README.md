# SD Forge Extra Euler Samplers

This repository provides additional Euler samplers for the Stable Diffusion (SD) Forge WebUI. These samplers enhance the capabilities of the SD Forge WebUI by offering more options for image generation.
The source for these new samplers come from projects created by [Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) and [licyk](https://github.com/licyk/advanced_euler_sampler_extension/tree/main).

## Features

- Additional Euler samplers integrated into the Forge WebUI.
  - Euler Max
  - Euler Negative
  - Euler Dy
  - Euler Dy Negative
  - Euler SMEA Dy
  - Euler SMEA Dy Negative
 
### Comparison

![comparison](https://github.com/user-attachments/assets/3d332ea5-14b4-4e86-af4f-a7c9f9938a0e)
*Model: leosamsHelloworldXL_helloworldXL70, Prompt: bustling town, sunset, street, cars, Steps: 30,
CFG: 5 (with DynamicThresholding), Size: 896 x 1152, Clip skip: 2, Seed: 3035751684*

## Installation

1. Clone this repository into the `extensions` directory of your WebUI installation:
    ```sh
    git clone https://github.com/yourusername/sd-forge-extra-euler-samplers.git
    ```
2. Restart the WebUI to load the new samplers.

## Usage

1. Open the WebUI.
2. Navigate to the sampler settings.
3. Select one of the newly added Euler samplers from the list.
4. Generate images as usual.

### Tips

- Most schedulers will work with these samplers for general generations and hires fix.
- The `Normal` and `Beta` schedulers do not work well with Adetailer. `Karras` is recommended for adetailer.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the Apache 2. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

If any of these are incorrect please let me know!

- Thanks to the developers of Automatic1111 and Forge.
- [Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) for the following sampler contributions:
  - Euler Negative
  - Euler Dy Negative
  - Euler SMEA Dy (basis of Euler SMEA Dy Negative as well)
- [licyk](https://github.com/licyk/advanced_euler_sampler_extension/tree/main) for the following sampler contributions:
  - Euler Max
- Special thanks to the contributors of this repository.
