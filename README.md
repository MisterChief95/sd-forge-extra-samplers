# SD Forge Extra Samplers

This repository provides additional Euler samplers to the Forge WebUI.
The source for these new samplers come from projects created by [Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler), [licyk](https://github.com/licyk/advanced_euler_sampler_extension/tree/main), and [Panchovix](https://github.com/Panchovix/stable-diffusion-webui-reForge/blob/70f68fd52cb70f8f64e18e6c8e775e35ddf70f67/ldm_patched/k_diffusion/sampling.py#L2844)

## Features

- Additional samplers integrated into the Forge WebUI.
  - Adaptive Progressive (Experimental)
  - Euler Max
  - Euler Negative
  - Euler Dy
  - Euler Dy Negative
  - Euler SMEA
  - Euler SMEA Dy
  - Euler SMEA Dy Negative
  - Heun Ancestral
  - Kohaku LoNyu Yog
  - Langevin Euler (Experimental)
  - Res Multistep
  - Res Multistep CFG++

Adds a new extension accordian titled "Extra Samplers" to allow adjusting certain samplers.
 
### Comparison

![comparison_1](https://github.com/user-attachments/assets/7f2aa2c1-1f13-42c3-bffe-531bf7a1ec1c)
![comparison_2](https://github.com/user-attachments/assets/92eedf75-d073-4894-a232-e666f0949865)
*Prompt: bustling town, sunset, street, cars  
Steps: 30, Schedule type: Karras, CFG scale: 5, Seed: 4103437930, Size: 832x1216, Model: leosamsHelloworldXL_helloworldXL70*

## Installation

1. Clone this repository into the `extensions` directory of your WebUI installation:
    ```sh
    git clone https://github.com/MisterChief95/sd-forge-extra-samplers.git
    ```
2. Restart the WebUI to load the new samplers.

## Usage

1. Open the WebUI.
2. Navigate to the sampler settings.
3. Select one of the newly added Euler samplers from the list.
4. Generate images as usual.

Because of how these Dy/SMEA samplers work, they are not guaranteed to function correctly in img2img scenarios (Hires fix, adetailer, etc).
If you get strange outputs or artifcating in these cases, try using a different sampler after generating the base image.

Euler Max seems to work fine in just about all scenarios from my own testing.

Adaptive Progressive and Langevin Euler are samplers of my own I've been experimenting with. The former uses Euler A-like sampling initially then switches to DPM++ 2M and finally a detailing stage as the sampling progresses.
Langevin Euler is an SDE sampler that uses Langevin dynamics to add noise as sampling progresses.

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
  - Kohaku LoNyu Yog
- [licyk](https://github.com/licyk/advanced_euler_sampler_extension/tree/main) for the following sampler contributions:
  - Euler Max
  - Euler SMEA
- [Panchovix](https://github.com/Panchovix/stable-diffusion-webui-reForge)
  - Res Multistep
  - Res Multistep CFG++
- Special thanks to the contributors of this repository.
