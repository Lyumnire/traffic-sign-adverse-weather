import urllib.request
import re
import os
import sys

# Tsinghua mirror is extremely fast and doesn't require proxy
MIRROR = "http://pypi.tuna.tsinghua.edu.cn/simple"

packages = ["timm", "pyyaml", "huggingface-hub", "safetensors", "fsspec"]

print("=== Downloading wheels offline via Tsinghua Mirror ===")
for pkg in packages:
    try:
        url = f"{MIRROR}/{pkg}/"
        print(f"Fetching metadata for {pkg} from {url}...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        # Specify 10 seconds timeout to prevent hanging
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            
        # Extract all .whl links using regex
        links = re.findall(r'href="([^"]+\.whl[^"]*)"', html)
        
        # Parse links and find matching wheel
        wheel_url = None
        for link in links:
            # clean up query params or fragments
            clean_link = link.split('#')[0]
            filename = os.path.basename(clean_link)
            
            # We want py3-none-any or win_amd64
            if "py3-none-any" in filename or "win_amd64" in filename:
                # If relative link, prepend package simple url
                if not clean_link.startswith("http"):
                    # Tsinghua mirror links are usually relative like "../../packages/..." or absolute paths
                    # Let's resolve it
                    if clean_link.startswith("/"):
                        wheel_url = "http://pypi.tuna.tsinghua.edu.cn" + clean_link
                    else:
                        wheel_url = url + clean_link
                else:
                    wheel_url = clean_link
                break
                
        if wheel_url:
            dest = os.path.basename(wheel_url.split('?')[0])
            if os.path.exists(dest):
                print(f"Already downloaded {dest}")
            else:
                print(f"Downloading {dest} from {wheel_url}...")
                urllib.request.urlretrieve(wheel_url, dest)
                print(f"[OK] Downloaded {pkg}")
        else:
            print(f"[Error] No compatible wheel found for {pkg}")
            
    except Exception as e:
        print(f"[Error] Failed to download {pkg}: {e}")
