# Windows desk: Bonsai 2 Heretic on the RTX 4090 and RTX 3090

These scripts set up the dual-GPU Windows desk (DESKTOP-CME56A7) to serve Bonsai 2 Heretic on
both cards, and the Mac to reach both through one port. Every step can run again: a second run
repairs or updates what the first one made.

## Build

Hosted runners build the engine. One build serves one CUDA architecture, so each card needs its
own build. Run the `V-Sekai Windows build` workflow with `arch` 89 (RTX 4090) or 86 (RTX 3090).
It uploads `dist-windows-sm<arch>`, and keeps it for 7 days. The sm_89 build of the
`V-Sekai MSI release` workflow (`dist-windows`) also works for the 4090.

## Deploy one card

From the Mac, in the repository root:

    nu v-sekai/desk/deploy.nu 3090              # newest unexpired build for the card
    nu v-sekai/desk/deploy.nu 4090 --run <id>   # the build of one workflow run
    nu v-sekai/desk/deploy.nu 4090 --keep-build # rewrite launcher, task and tunnel only

`deploy.nu` copies `install.nu` to the desk over SSH and runs it there. `install.nu` does the following:

1. It checks the model `C:/ninfer-all-models/bonsai2-27b-heretic.ninfer` against its sha256, and
   downloads it from the `chibifire/ninfer-bonsai2-heretic` dataset when it is missing.
2. It downloads the build and checks every file against `SHA256SUMS` and the architecture in
   `BUILDINFO.json`. Then it installs the build into `C:/ninfer-<card>`.
3. It writes `%LOCALAPPDATA%/V-Sekai/llm/serve-ninfer-<card>.cmd` and the logon task
   `\V-Sekai\llm-ninfer-<card>`. The launcher finds the card by name, in PCI bus order
   (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). Without that setting, CUDA's default order puts the 4090
   first, and the wrong card starts.
4. It starts the server and waits until the log says `listening on`.
5. It writes one stunnel section per installed card into `~/.ninfer-stunnel/stunnel.conf`, which
   the logon task `\V-Sekai\ninfer-stunnel` runs. The tunnel uses the desk's Bao serving cert, and
   only the Mac's agent identity can connect. It also updates the firewall rule.

| Card | Arch | Local port | Tunnel port | Mac port |
|---|---|---|---|---|
| 4090 | sm_89 | 8091 | 8443 | 18091 |
| 3090 | sm_86 | 8092 | 8444 | 18092 |

Then `deploy.nu` adds the Mac end of the tunnel (`com.chibifire.ninfer-tunnel`) and runs the
witnesses through it: a request without a key and one with a wrong key must get 401, a request
with the key must get 200, and it runs one temperature-0 code answer, printing decode speed and the
answer hash, and the image probe (red circle, blue square, "42").

Both servers read one API key, `%LOCALAPPDATA%/V-Sekai/llm/api-key.txt`, which is
`BONSAI_DESK_API_KEY` in the Mac's `~/.hermes/.env`. Do not print the body of `/v1/models`: it
echoes the server's command line, `--api-key` included.

## One port on the Mac

`router/` is a LiteLLM proxy on `127.0.0.1:18000`. Hermes uses it as the provider `bonsai`.

    nu v-sekai/desk/router/router.nu install   # then load the LaunchAgent from a terminal
    nu v-sekai/desk/router/router.nu check

The model `bonsai` sends each request to the least busy server that answers: desk 4090, desk
3090, Fedora 4090 or Fedora 3090. The desk dual-boots into Fedora, so only one pair is up at a
time, and the router skips a server that is down for 60 s. `bonsai-desk-4090`,
`bonsai-desk-3090`, `bonsai-fedora-4090` and `bonsai-fedora-3090` each select one card.
