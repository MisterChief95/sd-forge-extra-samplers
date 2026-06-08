# Overview

This repository provides additional samplers to the Forge Neo WebUI.

> [!CAUTION]
> Support for older versions of has been dropped since they are no longer maintained.
> Please ensure you are using the latest version of Forge Neo WebUI.

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
  - Langevin Euler (Experimental)
  - Res Multistep
  - Res Multistep CFG++
  - Res Multistep Ancestral
  - Res Multistep Ancestral CFG++
  - RES4LYF's `res` Samplers in both SDE and ODE flavors
    - res_2m
    - res_3m
    - res_2s
    - res_3s
    - res_4s
    - res_5s
    - res_6s
  - SSPRK3
 
- Additional Schedulers
  - Linear Log 

Adds a new extension accordian titled "Extra Samplers" to allow adjusting certain samplers.

## Installation

### Clone from Git

1. Navigate to the extension directory in your WebUI installation
1. Clone the repository:
    ```sh
    git clone https://github.com/MisterChief95/sd-forge-extra-samplers.git
    ```
1. Start WebUI

### Install from URL

1. Open the Extensions tab in the web UI.
2. Go to the "Install from URL" section.
3. Enter: `https://github.com/MisterChief95/sd-forge-extra-samplers.git` in the "URL for extension's git repository" box.
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

This extension builds on work from the broader Stable Diffusion and ComfyUI
sampler ecosystem. Thank you to the developers of AUTOMATIC1111, Forge, Forge
Classic, and the projects credited below.

- [Koishi-Star / Euler-Smea-Dyn-Sampler](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler):
  Euler Negative, Euler Dy, Euler Dy Negative, Euler SMEA Dy,
  and Euler SMEA Dy Negative.
- [licyk / advanced_euler_sampler_extension](https://github.com/licyk/advanced_euler_sampler_extension):
  Euler Max and Euler SMEA.
- [Panchovix / stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge):
  Res Multistep and Res Multistep CFG++.
- [comfyanonymous / ComfyUI](https://github.com/comfyanonymous/ComfyUI):
  Gradient Estimation, Extended Reverse Time SDE, Res Multistep,
  Res Multistep CFG++, Res Multistep Ancestral, and
  Res Multistep Ancestral CFG++.
- Euler Multipass: original implementation by
  [aria1th](https://github.com/aria1th), CFG++ implementation by
  [LaVie024](https://github.com/LaVie024), and final ComfyUI implementation by
  [catboxanon](https://github.com/catboxanon).
- [ClownsharkBatwing / RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF):
  RES4LYF beta sampler math and helper code used by the Forge RES4LYF sampler
  port. Files under `lib_es/res4lyf` include per-file AGPLv3 attribution
  headers, and `lib_es/res4lyf/PORTING_NOTES.md` documents Forge-specific
  changes.

If an attribution is incomplete or incorrect, please open an issue or pull
request so it can be corrected.
