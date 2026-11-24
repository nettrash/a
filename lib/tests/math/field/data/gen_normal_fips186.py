# Transcription of FIPS 186-4 (July 2013), Appendix D: the normal-basis representations of the
# NIST binary curves (D.1.3) and the rows of the conversion matrices (D.9, D.10). Each value is
# checked against the others in Python's integers (the normal basis given by D.10's theta, the
# polynomial values mapped through it), then written to normal_fips186.txt.
#
#   python3 gen_normal_fips186.py
# Values transcribed from FIPS 186-4, Appendix D.1.3 (curves), D.9 and D.10 (conversions).
F = {
163: dict(T=4, f=[163,7,6,3,0],
  theta="7 15169c10 9c612e39 0d347c74 8342bcd3 b02a0bef",
  root="3 e173bfaf 3a86434d 883a2918 a489ddbd 69fe84e1",
  values=[("K-163 Gx","2 fe13c053 7bbc11ac aa07d793 de4e6d5e 5c94eee8","0 5679b353 caa46825 fea2d371 3ba450da 0c2a4541"),
          ("K-163 Gy","2 89070fb0 5d38ff58 321f2e80 0536d538 ccdaa3d9","2 35b7c671 00506899 06bac3d9 dec76a83 5591edb2"),
          ("B-163 b","2 0a601907 b8c953ca 1481eb10 512f7874 4a3205fd","6 645f3cac f1638e13 9c6cd13e f61734fb c9e3d9fb"),
          ("B-163 Gx","3 f0eba162 86a2d57e a0991168 d4994637 e8343e36","0 311103c1 7167564a ce77ccb0 9c681f88 6ba54ee8"),
          ("B-163 Gy","0 d51fbc6c 71a0094f a2cdd545 b11c5c0c 797324f1","3 33ac13c6 447f2e67 613bf700 9daf98c8 7bb50c7f")]),
233: dict(T=2, f=[233,74,0],
  theta="149 9e398ac5 d79e3685 59b35ca4 9bb7305d a6c0390b cf9e2300 253203c9",
  root="0be 19b89595 28bbc490 038f4bc4 da8bdfc1 ca36bb05 853fd0ed 0ae200ce",
  values=[("K-233 Gx","172 32ba853a 7e731af1 29f22ff4 149563a4 19c26bf5 0a4c9d6e efad6126","0fd e76d9dcd 26e643ac 26f1aa90 1aa12978 4b71fc07 22b2d056 14d650b3"),
          ("K-233 Gy","1db 537dece8 19b7f70f 555a67c4 27a8cd9b f18aeb9b 56e0c110 56fae6a3","064 3e317633 155c9e04 47ba8020 a3c43177 450ee036 d6335014 34cac978"),
          ("B-233 b","066 647ede6c 332c7f8c 0923bb58 213b333b 20e9ce42 81fe115f 7d8f90ad","1a0 03e0962d 4f9a8e40 7c904a95 38163adb 82521260 0c7752ad 52233279"),
          ("B-233 Gx","0fa c9dfcbac 8313bb21 39f1bb75 5fef65bc 391f8b36 f8f8eb73 71fd558b","18b 863524b3 cdfefb94 f2784e0b 116faac5 4404bc91 62a363ba b84a14c5"),
          ("B-233 Gy","100 6a08a419 03350678 e58528be bf8a0bef f867a7ca 36716f7e 01f81052","049 25df77bd 8b8ff1a5 ff519417 822bfedf 2bbd7526 44292c98 c7af6e02")]),
283: dict(T=6, f=[283,12,7,5,0],
  theta="31e0ed7 91c3282d c5624a72 0818049d 053e8c7a b8663792 bc1d792e ba9867fc 7b317a99",
  root="3347f17 521fdabc 62ec1551 acf156fb 0bceb855 f174d4c1 7807511c 9f745382 add53bc3",
  values=[("K-283 Gx","503213f 78ca4488 3f1a3b81 62f188e5 53cd265f 23c1567a 16876913 b0c2ac24 58492836","3ab9593 f8db09fc 188f1d7c 4ac9fcc3 e57fcd3b db15024b 212c7022 9de5fcd9 2eb0ea60"),
          ("K-283 Gy","1ccda38 0f1c9e31 8d90f95d 07e5426f e87e45c0 e8184698 e4596236 4e341161 77dd2259","2118c47 55e7345c d8f603ef 93b98b10 6fe8854f feb9a3b3 04634cc8 3a0e759f 0c2686b1"),
          ("B-283 b","27b680a c8b8596d a5a4af8a 19a0303f ca97fd76 45309fa2 a581485a f6263e31 3b79a2f5","157261b 894739fb 5a13503f 55f0b3f1 0c560116 66331022 01138cc1 80c0206b dafbc951"),
          ("B-283 Gx","5f93925 8db7dd90 e1934f8c 70b0dfec 2eed25b8 557eac9c 80e2e198 f8cdbecd 86b12053","749468e 464ee468 634b21f7 f61cb700 701817e6 bc36a236 4cb8906e 940948ea a463c35d"),
          ("B-283 Gy","3676854 fe24141c b98fe6d4 b20d02b4 516ff702 350eddb0 826779c8 13f0df45 be8112f4","62968bd 3b489ac5 c9b859da 68475c31 5bafcdc4 ccd0dc90 5b70f624 46f49c05 2f49c08c")]),
409: dict(T=4, f=[409,87,0],
  theta="0dfa06b e206aa97 b7a41fff b9b0c55f 8f048062 fbe8381b 4248adf9 2912ccc8 e3f91a24 e1cfb395 0532b988 971c2304 2e85708d",
  root="0eb00f2 ea95fd6c 64024e7f 0b68b81f 5ff8a467 acc2b4c3 b9372843 6265c7ff a06d896c ae3a7e31 e295ec30 3eb9f769 de78bef5",
  values=[("K-409 Gx","060f05f 658f49c1 ad3ab189 0f718421 0efd0987 e307c84c 27accfb8 f9f67cc2 c460189e b5aaaa62 ee222eb1 b35540cf e9023746","1b559c7 cba2422e 3affe133 43e808b5 5e012d72 6ca0b7e6 a63aeafb c1e3a98e 10ca0fcf 98350c3b 7f89a975 4a8e1dc0 713cec4a"),
          ("K-409 Gy","1e36905 0b7c4e42 acba1dac bf04299c 3460782f 918ea427 e6325165 e9ea10e3 da5f6c42 e9c55215 aa9ca27a 5863ec48 d8e0286b","16d8c42 052f07e7 713e7490 eff318ba 1abd6fef 8a5433c8 94b24f5c 817aeb79 852496fb ee803a47 bc8a2038 78ebf1c4 99afd7d6"),
          ("B-409 b","021a5c2 c8ee9feb 5c4b9a75 3b7b476b 7fd6422e f1f3dd67 4761fa99 d6ac27c8 a9a197b2 72822f6c d57a55aa 4f50ae31 7b13545f","124d065 1c3d3772 f7f5a1fe 6e715559 e2129bdf a04d52f7 b6ac7c53 2cf0ed06 f610072d 88ad2fdc c50c6fde 72843670 f8b3742a"),
          ("B-409 Gx","15d4860 d088ddb3 496b0c60 64756260 441cde4a f1771d4d b01ffe5b 34e59703 dc255a86 8a118051 5603aeab 60794e54 bb7996a7","0ceacbc 9f475767 d8e69f3b 5dfab398 13685262 bcacf22b 84c7b6dd 981899e7 318c96f0 761f77c6 02c016ce d7c548de 830d708f"),
          ("B-409 Gy","061b1cf ab6be5f3 2bbfa783 24ed106a 7636b9c5 a7bd198d 0158aa4f 5488d08f 38514f1f df4b4f40 d2181b36 81c364ba 0273c706","199d64b a8f089c6 db0e0b61 e80bb959 34afd0ca f2e8be76 d1c5e9af fc7476df 49142691 ad303902 88aa09bc c59c1573 aa3c009a")]),
571: dict(T=10, f=[571,10,5,2,0],
  theta="452186b bf5840a0 bcf8c9f0 2a54efa0 4e813b43 c3d41496 06c4d27b 487bf107 393c8907 f79d9778 beb35ee8 7467d328 8274caeb da6ce05a eb4ca5cf 3c3044bd 4372232f 2c1a27c4",
  root="7940ffa ef996513 4d59dcbf e5bf239b e4fe4b41 05959c5d 4d942ffd 46ea35f3 e3cdb0e1 04a2aa01 cef30a3a 49478011 196bfb43 c55091b6 1174d7c0 8d0cdd61 3bf6748a bad972a4",
  values=[("K-571 Gx","26eb7a8 59923fbc 82189631 f8103fe4 ac9ca297 0012d5d4 60248048 01841ca4 43709584 93b205e6 47da304d b4ceb08c bbd1ba39 494776fb 988b4717 4dca88c7 e2945283 a01c8972","04bb2db a418d0db 107adae0 03427e5d 7cc139ac b465e593 4f0bea2a b2f3622b c29b3d5b 9aa7a1fd fd5d8be6 6057c100 8e71e484 bcd98f22 bf847642 37673674 29ef2ec5 bc3ebcf7"),
          ("K-571 Gy","349dc80 7f4fbf37 4f4aeade 3bca9531 4dd58cec 9f307a54 ffc61efc 006d8a2c 9d4979c0 ac44aea7 4fbebbb9 f772aedc b620b01a 7ba7af1b 320430c8 591984f6 01cd4c14 3ef1c7a3","44cbb57 de20788d 2c952d7b 56cf39bd 3e89b189 84bd124e 751ceff4 369dd8da c6a59e6e 745df44d 8220ce22 aa2c852c fcbbef49 ebaa98bd 2483e331 80e04286 feaa2530 50caff60"),
          ("B-571 b","2f40e7e 2221f295 de297117 b7f3d62f 5c6a97ff cb8ceff1 cd6ba8ce 4a9a18ad 84ffabbd 8efa5933 2be7ad67 56a66e29 4afd185a 78ff12aa 520e4de7 39baca0c 7ffeff7f 2955727a","3762d0d 47116006 179da356 88eeaccf 591a5cde a7500011 8d9608c5 9132d434 26101a1d fb377411 5f586623 f75f0000 1ce61198 3c1275fa 31f5bc9f 4be1a0f4 67f01ca8 85c74777"),
          ("B-571 Gx","303001d 34b85629 6c16c0d4 0d3cd775 0a93d1d2 955fa80a a5f40fc8 db7b2abd bde53950 f4c0d293 cdd711a3 5b67fb14 99ae6003 8614f139 4abfa3b4 c850d927 e1e7769c 8eec2d19","0735e03 5def5925 cc33173e b2a8ce77 67522b46 6d278b65 0a291612 7dfea9d2 d361089f 0a7a0247 a184e1c7 0d417866 e0fe0feb 0ff8f2f3 f9176418 f97d117e 624e2015 df1662a8"),
          ("B-571 Gy","37bf273 42da639b 6dccfffe b73d69d7 8c6c27a6 009cbbca 1980f853 3921e8a6 84423e43 bab08a57 6291af8f 461bb2a8 b3531d2f 0485c19b 16e2f151 6e23dd3c 1a4827af 1b8ac15b","04a3642 0572616c df7e606f ccadaecf c3b76dab 0eb1248d d03fbdfc 9cd3242c 4726be57 9855e812 de7ec5c5 00b4576a 24628048 b6a72d88 0062eed0 dd34b109 6d3acbb6 b01a4a97")]),
}

h=lambda s:int(s.replace(" ",""),16)
for m,d in F.items():
    f=sum(1<<e for e in d['f'])
    def mul(a,b):
        r=0
        while b:
            if b&1: r^=a
            b>>=1; a<<=1
            if a>>m&1: a^=f
        return r
    th=h(d['theta']); c=[th]
    for i in range(m-1): c.append(mul(c[-1],c[-1]))
    assert mul(c[-1],c[-1])==th
    def n2p(nv):  # FIPS normal string: a0 is the MSB
        r=0
        for i in range(m):
            if nv>>(m-1-i)&1: r^=c[i]
        return r
    assert n2p(h(d['root']))==2, m
    for name,p,n in d['values']:
        assert n2p(h(n))==h(p), (m,name)
    print(m,"ok")

out = ['# FIPS 186-4 (2013), Appendix D: for each binary field, the line "field m T exponents", then',
       '# "theta" (D.10: the top row of the normal-to-polynomial matrix, the normal element in the',
       '# polynomial basis), "root" (D.9: the second row from the bottom of the polynomial-to-normal',
       '# matrix, x in the normal basis), and "value name polynomial normal" for the coefficients and base',
       '# points of D.1.3. A normal-basis number has the coordinate a0 as its leading bit.',
       '# Transcribed by gen_normal_fips186.py, which checks each value against the others.']
for m, d in F.items():
    out.append('field %d %d %s' % (m, d['T'], ','.join(map(str, d['f']))))
    out.append('theta ' + d['theta'].replace(' ', ''))
    out.append('root ' + d['root'].replace(' ', ''))
    for name, p, n in d['values']:
        out.append('value %s %s %s' % (name.replace(' ', '_'), p.replace(' ', ''), n.replace(' ', '')))
open('normal_fips186.txt', 'w').write('\n'.join(out) + '\n')
