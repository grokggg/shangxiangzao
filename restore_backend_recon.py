#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""还原 backend_recon 三个原始文件。用法：python3 restore_backend_recon.py [输出目录]"""
import os,sys,struct,zlib,hashlib,base64
HERE=os.path.dirname(os.path.abspath(__file__))
img=open(os.path.join(HERE,"backend_recon_internal.png","backend_recon_internal.png"),"rb").read()
assert img[:8]==b"\x89PNG\r\n\x1a\n","not a PNG"
i=8; chunks=[]
while True:
    ln=struct.unpack(">I",img[i:i+4])[0]; t=img[i+4:i+8]; d=img[i+8:i+8+ln]
    if t==b"IEND": break
    if t==b"IDAT": chunks.append(d)
    i+=12+ln
    if i>=len(img): break
    if t==b"IEND": break
data=zlib.decompress(b"".join(chunks))
manifest_line,rest=data.split(b"\n===FILES===
",1)
manifest=json.loads(manifest_line)
outdir=sys.argv[1] if len(sys.argv)>1 else HERE
pos=0
for name,expect in manifest.items():
    nl=struct.unpack(">H",rest[pos:pos+2])[0]; pos+=2
    name2=rest[pos:pos+nl].decode(); pos+=nl
    vl=struct.unpack(">Q",rest[pos:pos+8])[0]; pos+=8
    body=rest[pos:pos+vl]; pos+=vl
    assert hashlib.sha256(body).hexdigest()==expect,f"checksum mismatch {name}"
    dst=os.path.join(outdir,name2.replace("/","_"))
    open(dst,"wb").write(body); print("restored",dst,len(body),"B")
print("ALL OK")
