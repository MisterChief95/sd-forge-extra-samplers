# SD Forge Extra Euler Samplers

This repository provides additional Euler samplers to the Forge WebUI.
The source for these new samplers come from projects created by [Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) and [licyk](https://github.com/licyk/advanced_euler_sampler_extension/tree/main).

## Features

- Additional Euler samplers integrated into the Forge WebUI.
  - Euler Max
  - Euler Negative
  - Euler Dy
  - Euler Dy Negative
  - Euler SMEA
  - Euler SMEA Dy
  - Euler SMEA Dy Negative
 
### Comparison

![comparison_1](https://github.com/user-attachments/assets/7f2aa2c1-1f13-42c3-bffe-531bf7a1ec1c)
![comparison_2](https://github.com/user-attachments/assets/92eedf75-d073-4894-a232-e666f0949865)
*Prompt: bustling town, sunset, street, cars  
Steps: 30, Schedule type: Karras, CFG scale: 5, Seed: 4103437930, Size: 832x1216, Model: leosamsHelloworldXL_helloworldXL70*

## Installation

1. Clone this repository into the `extensions` directory of your WebUI installation:
    ```sh
    git clone https://github.com/MisterChief95/sd-forge-extra-euler-samplers.git
    ```
2. Restart the WebUI to load the new samplers.

## Usage

1. Open the WebUI.
2. Navigate to the sampler settings.
3. Select one of the newly added Euler samplers from the list.
4. Generate images as usual.

### Tips

- Most schedulers will work with these samplers for general generations and Hires fix.
  - The SMEA schedulers do *not* work well for Hires fix/Adetailer. They cause smearing/deformation.
- The `Simple`, `Normal`, and `Beta` schedulers do not work well with Adetailer. `Uniform` and `Karras` are recommendeded for Adetailer.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the Apache 2. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

If any of these are incorrect please let me know!

- Thanks to the developers of Automatic1111 and Forge.
- [Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) for the following sampler contributions:
  - Euler Negative
  - Euler Dy
  - Euler Dy Negative
  - Euler SMEA Dy (Euler SMEA Dy Negative based on this)
- [licyk](https://github.com/licyk/advanced_euler_sampler_extension/tree/main) for the following sampler contributions:
  - Euler Max
  - Euler SMEA
- Special thanks to the contributors of this repository.
