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
  - Euler Multipass
  - Euler Multipass CFG++
  - Euler a Multipass
  - Euler a Multipass CFG++
  - Extended Reverse Time SDE
  - Gradient Estimation
  - Heun Ancestral
  - Kohaku LoNyu Yog
  - Langevin Euler (Experimental)
  - Res Multistep
  - Res Multistep CFG++
  - Res Multistep Ancestral
  - Res Multistep Ancestral CFG++
 
- Additional Schedulers
  - Linear Log 

Adds a new extension accordian titled "Extra Samplers" to allow adjusting certain samplers.

## Installation

## Clone from Git

1. Navigate to the extension directory in your WebUI installation
1. Clone the repository:
    ```sh
    git clone https://github.com/MisterChief95/sd-forge-preprocessor-nai-inpaint.git
    ```
1. Start WebUI

## Install from URL

1. Open the Extensions tab in the web UI.
2. Go to the "Install from URL" section.
3. Enter: `https://github.com/MisterChief95/sd-forge-preprocessor-nai-inpaint.git` in the "URL for extension's git repository" box.
4. Click "Install".
5. Restart WebUI

## Usage

1. Open the WebUI.
2. Navigate to the sampler settings.
3. Select one of the newly added Euler samplers from the list.
4. Generate images as usual.

### Important
- Not all samplers work well in every situation. Some will look poor when used for img2img/hires fix.
- Mix-and-match samplers to find the best combinations. A sampler might look bad with one scheduler but good with another!

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

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
- [Panchovix](https://github.com/Panchovix/stable-diffusion-webui-reForge) for the following sampler contributions:
  - Res Multistep
  - Res Multistep CFG++
- [comfyanonymous](https://github.com/comfyanonymous/ComfyUI) for the following sampler contributions:
  - Gradient Estimation
  - Extended Reverse Time SDE
  - Res Multistep
  - Res Multistep CFG++
  - Res Multistep Ancestral
  - Res Multistep Ancestral CFG++
- Euler Multipass
  - Original Implementation: [aria1th](https://github.com/aria1th)
  - CFG++ Implementation: [LaVie024](https://github.com/LaVie024)
  - Final ComfyUI implementation: [catboxanon](https://github.com/catboxanon)
- Special thanks to the contributors of this repository.
