//! The terms of Rademacher's series in multiple precision: fixed-point and floating numbers on
//! [`Number`]s, with bounds of their errors.
//!
//! A fixed-point number is an integer `X` for `X 2^-f`; a floating one a mantissa `m` and an
//! exponent `e` for `m 2^e`, the mantissa cut to a working number of bits `w` by truncation. Each
//! value carries a bound of its error: absolute, in units of `2^-f`, for fixed-point numbers, and
//! relative, in units of `2^-w`, for floating ones; each truncation adds one unit and each product
//! the sum of its factors' errors, so the bounds follow the computation step by step.
//!
//! `π` and `ln 2` come from tables of 16,384 bits ([`PI_LIMBS`], [`LN2_LIMBS`]), `π` beyond from
//! the Chudnovskys' series; the cosines of angles with small denominators from closed forms and the
//! half- and triple-angle formulas ([`cos_pi_algebraic`]), the others from Newton's method on
//! Chebyshev polynomials ([`cos_pi_newton`]) or from their series ([`cos_pi`]).

use super::super::super::{ByteOrder, Number};

/// The byte order of the working numbers: that of the limbs, which the arithmetic reads as it is.
const LE: ByteOrder = ByteOrder::LittleEndian;

/// `floor(π 2^16384)`, the least significant limb first, computed by Machin's formula with 128
/// guard bits: `π` to the precision of the series up to `n` of about `2 10^7`, without computing it.
const PI_LIMBS: [u64; 257] = [
    0xA969A7AAC50C06C2,
    0xE8EFD85561D99735,
    0xA607808419F8509E,
    0x4085F2A7CE77326E,
    0x7CDE3759CBEE7460,
    0xD65FECF16C223BDB,
    0x3372F0928D937E41,
    0x1DADF43E233F7061,
    0x58428D2A0C55F5EA,
    0x1462B17423820E00,
    0x095BBF00AD19489D,
    0x47848A0B5692B285,
    0x7F1524C369CB7492,
    0xFC8883A0C1C7B6A3,
    0xCF62A1F25B8D2646,
    0xE44B476A3D816250,
    0xD81E799E86854DC7,
    0x654F3B1D41CD2105,
    0xC742F442EF6ABBB5,
    0xDB6C4F15FACB4FD0,
    0x45EEE2B6A3AAABEA,
    0x42105D14203E13E0,
    0x5D4A14D9E864B7E3,
    0xFE28ED61EE7C3C73,
    0x58EBF2EF34C6FFEA,
    0x8FD948E46DBC3128,
    0xCDB30AEB532E3054,
    0x9B83C3FF1AC24696,
    0x57F584A51B227263,
    0x96DEDFA17858BA99,
    0x0E358829C7E61FD6,
    0x11ED935F16681281,
    0xF837889A97E32D77,
    0x95F7997E623D7DA8,
    0x9B540B19875FA099,
    0x319EE9D5C021B8F7,
    0x40685A323C2AB4B3,
    0xA02369B9655ABB50,
    0x648B1EAF19BDF0CA,
    0x9CAB5CABB2F3846E,
    0xEECC86BC60622CA7,
    0xD59E9E0BCBAADE14,
    0xB5735C904C70A239,
    0x0334FE1EAA0363CF,
    0x13CCA830EB61BD96,
    0xECDD4775619F1510,
    0x1521B62829076170,
    0x2AEF7DAD5B6E2F84,
    0xEECEA50FDB2F953B,
    0x6558218568AB9802,
    0xC332DDEFBE6C5AA5,
    0xB17F37D1018CFF28,
    0xE3BC4595A67BC883,
    0xEBADFE6EEAC31F66,
    0x5223A708F71312B6,
    0x1939260F19C27960,
    0xF2F74EA7361D2B3D,
    0xA5FC3C531E0A2DF4,
    0xC6150EBA94E2EA78,
    0x2E4CC9789C10B36A,
    0x803E89D65266C825,
    0x2965DCB999E71D0F,
    0x771FE71C4E3D06FA,
    0x860E5E0A5A3E2AB3,
    0x86E34570EAE96FB1,
    0x9EBABF2C1E153C6E,
    0xF28FE6ED97F1FBFA,
    0x3C11183B5924A509,
    0x043556F1D7A3C76B,
    0x81AC77D65F11199B,
    0x71DFF89E10314E55,
    0x50940002133AE4DD,
    0x12A14D432A65C451,
    0xDD433B3724C2BA16,
    0xEF1C18473215D908,
    0x63094366C464C3D2,
    0xEC7AEC3ADB851DFA,
    0xBC9F9ABCE94B7D8C,
    0x7A5847187408DA17,
    0x501ADDE69F84CD87,
    0x5512721F2E6B7124,
    0xB38BAE12DCCF3F2E,
    0xDE9A771FD9930810,
    0x5679B072BCAF89AF,
    0x2CB8129024977C79,
    0x6F420D03F60A04BF,
    0x4E548B384F6DB908,
    0x3280BBA1183EB331,
    0xE238CD993BEA0E2F,
    0xECC8C73EA4751E41,
    0xA94461460FD0030E,
    0xC8B576349AF3DDA7,
    0x3AE5E58137C2DADC,
    0x9432477322F54701,
    0xD19113F97CA92FF6,
    0x25837A58DC0921BD,
    0xAE0CF51A3CB574B2,
    0xF01C1F040200B3FF,
    0xB03ADA37F0500C0D,
    0x8E7D44EC5716F2B8,
    0x3E07841C7FDEAE5C,
    0xB79C5305AA500737,
    0x687F358452A0E286,
    0x3C9718146B6A70A1,
    0x09686B3F3EBAEFC9,
    0x6382E9C6021ECC5E,
    0x4CDD20868470EB26,
    0x4D2D38E6F0255DC1,
    0xA1D29C07EFE830F5,
    0x6B8FE4D699F73FD6,
    0x3F54989A5B429D65,
    0xA0591340E4183A3E,
    0x193602A575094C29,
    0x5664526CC2B19EE1,
    0xECAA8C71699A17FF,
    0x9CEE60B88FEDB266,
    0xAD6EA6B049A7DF7D,
    0xDB75092EC4192623,
    0x4B7A70E9B5B32944,
    0x08BA47996E85076A,
    0x53B02D5DA99F8FA1,
    0xFF34052EC5855664,
    0xB6636521E7B9F9B6,
    0xF296EC6B2A0DD915,
    0x08BA6FB5571BE91F,
    0x2B60A47681E67400,
    0xD60F573FBC9BC6E4,
    0x1B5100529A532915,
    0xB3472DCA7B14A94A,
    0x11C819684E734A41,
    0x832603766295CFA9,
    0x207D5BA202E5B9C5,
    0x78C14389D95A537F,
    0x5563911D59DFA6AA,
    0x2464369BF009B91E,
    0x226800BB57B8E0AF,
    0x00250E2D2071B35E,
    0xD6411BD3AE1E7E49,
    0xEBCDAF0C7B3E89A0,
    0xC75442F5FB9D35CF,
    0xE6AD206577B5FA86,
    0x93CC7314211A1477,
    0x165FA26680957705,
    0x7CC43B81D2ADA8D9,
    0xB4A84FE0FD13E0B7,
    0x18ACF3D6CE89E299,
    0xE5A0CC0FB56F74E8,
    0xEA752DFE8B021FA1,
    0x2F2F2218BE0E1777,
    0xD1CFF191B3A8C1AD,
    0x4FAD5EA0688FC31C,
    0x8888B812900DF01C,
    0x8FF6E2FBF2122B64,
    0x8E3C5B2F8E7594B7,
    0xD08ED1D0AFC725E0,
    0xEAAD8E716B93D5A0,
    0x95DBDA4DAE909198,
    0xD07E9EFE2BF11FB4,
    0xEF20CADA36774C01,
    0x62FB1341CEE4C6E8,
    0xE1DDF2DAA4CB7E33,
    0xD28E49BC4BFB9790,
    0x9A53E479B6F84565,
    0x4AFCB56C2DD1D35B,
    0x10FA3D98FD2183B8,
    0xE1FFA35DB8F011A0,
    0x695B27B0BBCA58C8,
    0xAC6732C68C4F5573,
    0xD542A8F6287EFFC3,
    0x1A87562EDF1769DB,
    0x9E5C57BBCA6F8CA0,
    0x53317B483E00DF82,
    0x323DB5FAFD238760,
    0x2F501EC8AD0552AB,
    0x3C7516DFFD616B15,
    0x8EA5E9F8DB3222F8,
    0x679F25FEFB1FA3CC,
    0xD6A100C6402C7279,
    0x5579C0BD1A60320A,
    0xD20B5F39B9D3FBDB,
    0xC0CBA85745C8740F,
    0x660F2807192E4BB3,
    0xBEE3D004DE334AFD,
    0xCC814544AF5EBD09,
    0x6DFC511F9B30952C,
    0x1339B2EB3B52EC6F,
    0x68FB6FAF3E6C53B5,
    0x5E5C9EC2196A2463,
    0xC1A94FB6409F60C4,
    0x976CE0BD04C006BA,
    0xE3FE501AB6794C3B,
    0x25D479D8F6E8DEF7,
    0x075372C980991B7B,
    0xDB0FEAD349F1C09B,
    0x37D0D724D00A1248,
    0x1BFEDF72429B023D,
    0x4ED3AA62363F7706,
    0x401A449F56C16AA6,
    0xE06F75D885C12073,
    0x7D84A5C33B8B5EBE,
    0x8CEE8619456F9FB4,
    0x66CA593E82430E88,
    0xA1F1651D39AF0176,
    0xBA3BF0507EFB2A98,
    0x6EEF0B6C137A3BE4,
    0x960FA728AB5133A3,
    0x6A51A0D2D8542F68,
    0x670C9C61ABD388F0,
    0x21C66842F6E96C9A,
    0x69C8F04A9E1F9B5E,
    0x2E0B4482A4842004,
    0x0F6D6FF383F44239,
    0x23893E81D396ACC5,
    0xDC262302EB651B88,
    0xEF845D5DE98575B1,
    0x487CAC605DEC8032,
    0x61D809CCFB21A991,
    0xC4BFE81B66282193,
    0x3B8F48986B4BB9AF,
    0x7A32538128958677,
    0xAFD6BA336C24CF5C,
    0xCE5C3E169B87931E,
    0x2BA9C55D741831F6,
    0xB3EE1411636FBC2A,
    0xA15486AF7C72E993,
    0xB4CC5C341141E8CE,
    0x55CA396A2AAB10B6,
    0x5748986263E81440,
    0xE65525F3AA55AB94,
    0x78AF2FDA55605C60,
    0xD71577C1BD314B27,
    0x6C9E0E8BB01E8A3E,
    0x8E79DCB0603A180E,
    0xCA417918B8DB38EF,
    0xC5D1B023286085F0,
    0x9C30D5392AF26013,
    0x7B54A41DC25A59B5,
    0x718BCD5882154AEE,
    0x0D95748F728EB658,
    0xA458FEA3F4933D7E,
    0x636920D871574E69,
    0x0801F2E2858EFC16,
    0x24A19947B3916CF7,
    0xBA7C9045F12C7F99,
    0xB8E1AFED6A267E96,
    0x2FFD72DBD01ADFB7,
    0xD1310BA698DFB5AC,
    0x9216D5D98979FB1B,
    0x3F84D5B5B5470917,
    0xC0AC29B7C97C50DD,
    0xBE5466CF34E90C6C,
    0x452821E638D01377,
    0x082EFA98EC4E6C89,
    0xA4093822299F31D0,
    0x13198A2E03707344,
    0x243F6A8885A308D3,
    0x0000000000000003,
];

/// `floor(ln 2 2^16384)`, the least significant limb first, from `Σ 1 / (k 2^k)` in integers with 128
/// guard bits: `ln 2` for the reduction of the first term's exponential up to the precision of
/// [`PI_LIMBS`], and the fraction of every value of the terms on limbs.
pub(super) const LN2_LIMBS: [u64; 256] = [
    0xDD9C92F002CA30F8,
    0x87B4C29E7FB208A9,
    0x613EE2AC34E4E9E7,
    0x8DE2AA9176BDB500,
    0x4DBFE4CEBD7FE74B,
    0x66840367AE0B74CB,
    0xFED51F626393CCC5,
    0x990E84C33202ECC9,
    0x054D8B71FC0C0D10,
    0x119F38BD9D90D43E,
    0x8CEE2EC9AF2D8731,
    0x8E0352BB66B37977,
    0x12C3E3F53DB5A720,
    0x82997D87B22199E9,
    0x7AEFCB7AF226FE13,
    0x6FC5EE5748041D64,
    0xBDDA0B625F39C4C8,
    0x99AB4C95D1A1ED94,
    0x40257B2B7F308EB7,
    0xB3D1FEEDCB09F14C,
    0x57FBEFFA0345B5AA,
    0xE7D993D899572381,
    0xB6DBF8BC4E6D07F9,
    0xA44A4D5D7E8DAB6B,
    0xF32018E14FE46C75,
    0x932247FF56563467,
    0x2B9F73AD3DDA7560,
    0x6DA24640BBE2BA0F,
    0x94CCC4CDFEFC08EA,
    0x343F671D4877FC71,
    0x607B1B85535B4CE9,
    0x43FD280FCE7DF51D,
    0xEFADE356ED90B7A9,
    0xFA220BBDEC995C53,
    0x7D06BDA5B72AA613,
    0xEE11689688D0FD96,
    0x0B82E686CE94A838,
    0x03D0604A998E0195,
    0xA3734AD2CC114BFF,
    0x95AF931CF8317D02,
    0xED5F3D14AF707BB0,
    0xEEC90443850ACA3B,
    0xA9024784BE09F609,
    0x1179057FC450DCA7,
    0xD20A751FCC43A7C4,
    0xA203F76ED0E79070,
    0x6410110BD615E257,
    0x57FFCD4A70AA1EDF,
    0xFA4D768EE2FE5271,
    0x1B2F45A16299E5AC,
    0xCB9856B29FADC8EC,
    0x6A3141F601E3BD2B,
    0xB4AF621A8A555E27,
    0x5928DCDB041E9325,
    0x018D2B3101223478,
    0xAFAE7DAC1D5387BF,
    0xA55783CE134A42BB,
    0x2BA8C6120DA6A649,
    0x738AF20F3CCF6D58,
    0x58A59908F1670776,
    0xBE8FDB6792689FF1,
    0x6D105839F0319222,
    0x680B77F6CAB5ED37,
    0xFA4507BA0026761A,
    0x3C7C334B5EFDB4C6,
    0x582DC46AE726E49E,
    0x98570B152CEC2FBD,
    0x9C45E2715D3D15C2,
    0xA527BD4447ADB7D1,
    0x7C847E90A373F5A9,
    0x03F81FC6473930FF,
    0x20A7043A0742281B,
    0x414CBC1BB5F505D2,
    0x7A6ED2658ADEA5EE,
    0xFD036381A74342DA,
    0x8AEEDE384FA88B87,
    0x9E7F586217AEB406,
    0xD66AE99D7EFBEA3D,
    0xC3CE35B948CB5253,
    0x1A4C99C9A092C106,
    0xA769F887324B53DE,
    0x0C36312177D1B7B1,
    0xC43962E112EE6799,
    0x4694870E71C194CB,
    0x5585F6DDD13A0595,
    0x6C5501316DED1F3E,
    0x95DBBD18C4EF702D,
    0x2268A5E0C92F5181,
    0xE2C73E6FDC29696E,
    0xC901AFF64A09C45B,
    0x7533E747D9047F55,
    0x9DB25DA9BECCB4C2,
    0x15A2832291FF14FC,
    0xC1684223D7C6E681,
    0x3E08EB679C1DB7FF,
    0xDD826C5D3770067E,
    0x88255E1C0839CBCB,
    0x817642C6F224F70F,
    0x7F4CE0B6FFD8B947,
    0x9AEE5B60899D5224,
    0xDF1FFC2E2288F8CA,
    0x7EE236FDC33BC8E7,
    0x0116E6B65AEB3BE7,
    0x5709468D78EBD2DA,
    0xF60A3EB81747F87E,
    0xEF1FD9678CC0B5E9,
    0x5A8B7808DF43BD8E,
    0x341B06E211977B07,
    0x2B661CD85063034E,
    0xDD617335DFDD424F,
    0xC7BBA02811376090,
    0xD6F7E1B9E1903D71,
    0x7C7CD17AF804D933,
    0x27A900B1AF5E75F8,
    0xB01B2AE0E98E0E15,
    0x843BFA1873F0C446,
    0x5B2238677A2BFBBD,
    0x0FA8601CF6868A05,
    0x93435B9C277736A7,
    0x5A66203D62FD1E70,
    0xB16EA8911AFBF1AE,
    0x77CF0656907FB9B1,
    0xAC5E3701D7D77253,
    0xFEA79D89D5C5ED40,
    0x144E67EBE9B42698,
    0x971CC90B5518569F,
    0xA4B8D1ECF7567EB0,
    0xCBB9AC407DDB6C13,
    0xD571EC6C1366A992,
    0x435A0CE134C2838F,
    0xD2AAE89CCC3B76FC,
    0xEE94E62F110A6783,
    0xBAF86856CCD3C3B6,
    0x988012E8314186ED,
    0x8A886EB3C87B7295,
    0xE30219C8AA9CE884,
    0x1D822DD6E2F76797,
    0x852BE3E8FC99F14D,
    0x5A9139DB14EFCC30,
    0x347F8304D889659E,
    0x4012A82962C59CAB,
    0xDF59EAE051707062,
    0xB12DFAB414451579,
    0x80BC423433562E94,
    0xAF6D605871EF7AFB,
    0x1A356B2A73B7EAAD,
    0xF91096AC3195220A,
    0xC95F260FD10036F9,
    0xF9EEEEA98A2400CA,
    0x6DC085A98AC8D8CA,
    0x6808292057FD99B6,
    0x624F14A51A4A026B,
    0x73E5B5C1585318E7,
    0xC12963B0FF01EAAB,
    0x2B552879A6168695,
    0x8FD9405789F45681,
    0xF2D89D2A4B183527,
    0x1BDA1F85EF6FDBF2,
    0x1AA4FB42B9A3DEF4,
    0xD6CCE1DAA5053701,
    0xAC14B958784934B8,
    0x12B5E8C202461069,
    0xA3D091F656658154,
    0xD162AF053B1751F3,
    0x846532E4B9694EDA,
    0xB779DFE49D7307CC,
    0xAD8A43DC4212B210,
    0x6FE51A8CFAA72EF2,
    0x88D586554E2A0E8A,
    0x20CDB5CCB3DB2392,
    0x14F0CD976EA354BB,
    0x471BF4F445F0A88A,
    0x0F023B220224FCD8,
    0xF11785903155BBD9,
    0x33CE3573FACC5FDC,
    0x154C60320E2FF793,
    0x53DAEC3F64F1B783,
    0xDB4A9316F281501E,
    0xBE2EC92156C9F949,
    0x0CA8F58D94F0341C,
    0x554B03D7D2874A00,
    0xFB0C75DF5497E00C,
    0xEE6E0850ECA42D06,
    0x364F5B8AEF22EC2F,
    0x897A39CE78B63C9F,
    0x52AB33161E238438,
    0x062B1A63A6C4C60C,
    0x3EA8449FE8F70EDD,
    0x6425A41526FAC51C,
    0xC5E5767DF95884E0,
    0xC0B1B31D8A0E23FA,
    0x85DB6AB03A49BD0D,
    0x175EB4AFC8DAADD8,
    0xF07AFFF3A892374E,
    0x8F6826250DEA891E,
    0xCECB72F19C38339D,
    0x5F6F7CEBAC9F45AE,
    0x6C472096E76115C0,
    0x972CD18BFBBD9D42,
    0x0AB111BBBD67C724,
    0x473826FDA0C238B9,
    0x61C1696DD24AAEBD,
    0x156E0C292413D5E3,
    0x95184460DC4E7487,
    0xD7622658901E646A,
    0xEF2F0CE2D7373958,
    0x2AC5B61CC4E9207C,
    0x57339BA2BEBA7D05,
    0x0060E49908391A0C,
    0x621363196AF50302,
    0x05C128D53D0BD2F9,
    0x36E02B20CEE886B9,
    0x0BBB16FAF3D949F2,
    0x422183EDC9942109,
    0x5E9222B88C66D3C5,
    0x61AFFD446B1CA3CF,
    0x268A5C1F9538B982,
    0x8D6F5177FBCF0755,
    0xA17293D1228A4EF9,
    0x44A02554731CDC8E,
    0x96D4E6D330AF889B,
    0x5570B6C68F969834,
    0x7598A1951AE273EE,
    0x4D162DB3B365853D,
    0x5F50B5185064C18B,
    0x078F735D1B2DB31B,
    0xAE313CDB6C606CB1,
    0x955D5179B1E17B9D,
    0x0C480A5417350D2C,
    0x074DB6015CFE7AA3,
    0x6A9C7F8A5E148E82,
    0x25669B333564A337,
    0x4C1A1E0BD1D6095D,
    0xCCCC4E659393514C,
    0xC943E732B479CD33,
    0x17460775DB8990E5,
    0x7D2E23DE1400B396,
    0xEE569D6DFC1EFA15,
    0x610D30F88FE551A2,
    0x07F4CA11FB5BFB90,
    0xDA2D97C50F3FD5C6,
    0x655FA1872F20E3A2,
    0xF5DFA6BD38303248,
    0x72CE87B19D6548CA,
    0x256FA0EC7657F74B,
    0xB9EA9BC3B136603B,
    0x1ACBDA11317C387E,
    0x3E96CA16224AE8C5,
    0x27573B291169B825,
    0xED2EAE35C1382144,
    0x559552FB4AFA1B10,
    0xE7B876206DEBAC98,
    0x8A0D175B8BAAFA2B,
    0x40F343267298B62D,
    0xC9E3B39803F2F6AF,
    0xB17217F7D1CF79AB,
];

/// The fraction of `π`, [`PI_LIMBS`] without its integer part.
pub(super) const PI_TABLE: &[u64] = PI_LIMBS.split_last().expect("not empty").1;

/// The bits of [`PI_LIMBS`] below the point.
const PI_BITS: u64 = 16384;

/// The number of the limbs `limbs`, the least significant first.
fn of_limbs(limbs: &[u64]) -> Number {
    Number::new(limbs.iter().flat_map(|l| l.to_le_bytes()).collect(), LE)
}

/// A count of bits as a `usize`, for a shift: on 32-bit targets the precisions of the longest
/// series pass `usize::MAX`, numbers that cannot be allocated, refused as their allocation would be.
pub(super) fn shift_count(k: u64) -> usize {
    usize::try_from(k).expect("capacity overflow")
}

/// `x 2^k` for `k` of either sign, rounded down.
pub(super) fn shift(x: &Number, k: i64) -> Number {
    if k >= 0 {
        x << shift_count(k as u64)
    } else {
        x >> shift_count(k.unsigned_abs())
    }
}

/// The longest precision of `π` that [`PI_CACHE`] keeps: `2^22` bits, 512 KiB, the series of `n` up
/// to about `10^12`.
const PI_CACHE_BITS: u64 = 1 << 22;

/// `π` beyond the table, for the series of `n` above about `2 10^7`: the longest computed so far, as
/// `(f, floor(π 2^f) less at most 4)`, for the process; filled by [`pi`] at its first precision
/// above the table and replaced by a longer one, never past [`PI_CACHE_BITS`]. The partition
/// numbers do not depend on it: a value cut from a longer one stays within the same bound.
static PI_CACHE: std::sync::Mutex<Option<(u64, Number)>> = std::sync::Mutex::new(None);

/// `floor(π 2^f)` less at most 4: from the table up to 16,384 bits, by Chudnovsky's series beyond,
/// kept in [`PI_CACHE`] and cut from there for shorter precisions (`floor(x 2^-d)` of a value within
/// 4 below is within `1 + 4 2^-d <= 3` below for `d >= 1`).
pub(super) fn pi(f: u64) -> Number {
    if f <= PI_BITS {
        return of_limbs(&PI_LIMBS) >> shift_count(PI_BITS - f);
    }
    if let Ok(cache) = PI_CACHE.lock()
        && let Some((bits, v)) = cache.as_ref()
        && *bits >= f
    {
        return v >> shift_count(bits - f);
    }
    let v = chudnovsky(f);
    if f <= PI_CACHE_BITS
        && let Ok(mut cache) = PI_CACHE.lock()
        && cache.as_ref().is_none_or(|(bits, _)| *bits < f)
    {
        *cache = Some((f, v.clone()));
    }
    v
}

/// `π 2^f` within 4 below, by the Chudnovskys' series (1988)
/// `1/π = 12 Σ (-1)^j (6j)! (13591409 + 545140134 j) / ((3j)! (j!)^3 640320^(3j + 3/2))`, summed by
/// binary splitting: each term adds more than 47.11 bits, so `f / 47 + 2` terms leave a tail below
/// `2^-f`; then `π = 426880 √10005 Q / (13591409 Q + R)`, a square root and a quotient rounded down.
fn chudnovsky(f: u64) -> Number {
    let terms = f / 47 + 2;
    let (_, q, r) = split(1, terms + 1, false);
    let g = f + 8;
    let root = (Number::new_from_u64(10005, LE) << shift_count(2 * g)).isqrt();
    let den = &(&q * 13_591_409u32) + &r;
    // `Q` and the denominator have nearly twice the bits that the quotient needs: both cut to
    // `f + 64` bits, which moves it by less than `2^-(f + 62)` relatively.
    let cut = shift_count(den.bits().saturating_sub(f + 64));
    let (q, den) = (&q >> cut, &den >> cut);
    let num = &(&root * &q) * 426_880u32;
    (&num / &den) >> 8usize
}

/// The binary splitting of Chudnovsky's series: `(P, Q, R)` of the terms from `a` to `b - 1`, `P`
/// only when `need_p` (the whole series does not need it).
fn split(a: u64, b: u64, need_p: bool) -> (Number, Number, Number) {
    if b - a == 1 {
        // `p = -(6a - 5)(2a - 1)(6a - 1)`, `q = 10939058860032000 a^3`, `r = p (545140134 a +
        // 13591409)`: in `i128`s for `a < 2^20`, below `2^67`, `2^114` and `2^117`.
        if a < 1 << 20 {
            let x = i128::from(a);
            let p = -(6 * x - 5) * (2 * x - 1) * (6 * x - 1);
            let q = 10_939_058_860_032_000 * x * x * x;
            let r = p * (545_140_134 * x + 13_591_409);
            return (
                Number::new_from_i128(p, LE),
                Number::new_from_i128(q, LE),
                Number::new_from_i128(r, LE),
            );
        }
        let a = Number::new_from_u64(a, LE);
        let p = -(&(&(&(&a * 6u32) - 5u32) * &(&(&a * 2u32) - 1u32)) * &(&(&a * 6u32) - 1u32));
        let q = &(&(&a * &a) * &a) * 10_939_058_860_032_000u64;
        let r = &p * &(&(&a * 545_140_134u32) + 13_591_409u32);
        return (p, q, r);
    }
    let m = a + (b - a) / 2;
    let (p1, q1, r1) = split(a, m, true);
    let (p2, q2, r2) = split(m, b, need_p);
    let r = &(&q2 * &r1) + &(&p1 * &r2);
    let p = if need_p {
        &p1 * &p2
    } else {
        Number::new_from_u8(0, LE)
    };
    (p, &q1 * &q2, r)
}

/// A positive floating number `m 2^e`.
#[derive(Clone)]
pub(super) struct Float {
    /// The mantissa.
    pub(super) m: Number,
    /// The exponent.
    pub(super) e: i64,
}

impl Float {
    /// The fixed-point `x 2^-f`.
    pub(super) fn of_fixed(x: Number, f: u64) -> Float {
        Float {
            m: x,
            e: -(f as i64),
        }
    }

    /// The number cut to `w` bits of mantissa, and whether it lost any (a unit of `2^-w` of
    /// relative error, as the mantissa keeps its top bit).
    pub(super) fn cut(mut self, w: u64) -> Float {
        let bits = self.m.bits();
        if bits > w {
            let k = bits - w;
            self.m = &self.m >> shift_count(k);
            self.e += k as i64;
        }
        self
    }

    /// The nearest double, about: the top 64 bits rounded, within `1.0001 u` relatively.
    pub(super) fn to_f64(&self) -> f64 {
        let bits = self.m.bits();
        if bits > 64 {
            let k = bits - 64;
            (&self.m >> shift_count(k)).to_f64() * pow2(self.e + k as i64)
        } else {
            self.m.to_f64() * pow2(self.e)
        }
    }

    /// `log2` of the number, about.
    pub(super) fn log2(&self) -> f64 {
        let bits = self.m.bits();
        let top = if bits > 60 {
            (&self.m >> shift_count(bits - 60)).to_f64().log2() + (bits - 60) as f64
        } else {
            self.m.to_f64().abs().log2()
        };
        top + self.e as f64
    }

    /// The product, cut to `w` bits.
    pub(super) fn mul(&self, o: &Float, w: u64) -> Float {
        Float {
            m: &self.m * &o.m,
            e: self.e + o.e,
        }
        .cut(w)
    }

    /// The sum of two positive numbers, cut to `w` bits.
    pub(super) fn add(&self, o: &Float, w: u64) -> Float {
        let e = self.e.min(o.e);
        let (a, b) = if self.e >= o.e { (self, o) } else { (o, self) };
        // `b` below a quarter of a unit of `a`'s `w` bits: dropped.
        let top_a = a.e + a.m.bits() as i64;
        let top_b = b.e + b.m.bits() as i64;
        if top_b + w as i64 + 2 < top_a {
            return a.clone().cut(w);
        }
        Float {
            m: &shift(&a.m, a.e - e) + &shift(&b.m, b.e - e),
            e,
        }
        .cut(w)
    }

    /// The quotient by `d`, cut to `w` bits: a unit of `2^-w` lost to the truncation.
    pub(super) fn div(&self, d: &Float, w: u64) -> Float {
        // Enough bits in the dividend for `w + 2` bits of quotient.
        let k = (w + 2 + d.m.bits()).saturating_sub(self.m.bits());
        Float {
            m: &(&self.m << shift_count(k)) / &d.m,
            e: self.e - k as i64 - d.e,
        }
        .cut(w)
    }

    /// The fixed-point `floor(x 2^f)`.
    pub(super) fn to_fixed(&self, f: u64) -> Number {
        shift(&self.m, self.e + f as i64)
    }
}

/// `Σ_(j<t) c_j z^j` for the fixed-point `z = Z 2^-f`, `0 <= z < 1/2`, with `c_0 = 1` and
/// `c_j = c_(j-1) s / d(j)`, `s = -1` when `alternate` and 1 otherwise, `d(j) >= 2`, by rectangular
/// splitting (Smith, 1989): the powers `z^2` to `z^m`, `m` about `√t`, then the blocks of `m` terms
/// from the top, `G = z^0 + s (z^1 + s (z^2 + ... (z^m G') / d) / d) / d`, a product, `m`
/// multiplications by the small `d`s and a division by their product each. Returns the value in fixed point and a bound of its absolute error in units of
/// `2^-f`, for `z` within `ez` units, not counting the tail of the series past `t` terms.
///
/// The powers' errors are at most `i (ez + 1)` for `z^i` (a square `z^(2j)` within twice `z^j`'s
/// and one). In a block, the product `z^m G'` is within
/// `2 e_m + e(G') / 2^m + 1`, as `G' < 2` and `z^m < 2^-m`, and each step `z^(i-1) + q / d` adds
/// `e_(i-1) + 1` to a previous error divided by `d >= 2`: a block's error is below
/// `3m (ez + 1) + 2.5` plus half of the one above it, and the whole below twice that. The tail past
/// `t` terms is counted by the callers (see [`terms`]).
pub(super) fn series(
    z: &Number,
    f: u64,
    ez: f64,
    t: u64,
    d: impl Fn(u64) -> u64,
    alternate: bool,
) -> (Number, f64) {
    let one = Number::new_from_u8(1, LE) << shift_count(f);
    let m = (t as f64).sqrt().ceil().max(1.0) as u64;
    let mut powers = Vec::with_capacity(m as usize + 1);
    powers.push(one.clone());
    powers.push(z.clone());
    for i in 2..=m as usize {
        // The even powers as squares, which take fewer products of limbs.
        let p = if i % 2 == 0 {
            &(&powers[i / 2] * &powers[i / 2]) >> shift_count(f)
        } else {
            &(&powers[i - 1] * z) >> shift_count(f)
        };
        powers.push(p);
    }
    let blocks = t.div_ceil(m);
    let mut g = Number::new_from_u8(0, LE);
    for b in (0..blocks).rev() {
        // `D G_b = T_(m-1) d_m + s^m z^m G'`, `D` the product of the block's `d`s, with
        // `T_0 = 1`, `T_i = T_(i-1) d_i + s^i z^i`: multiplications by the small `d`s, cheaper than
        // divisions, and one division by `D`.
        let mut acc = one.clone();
        let mut den = Number::new_from_u8(1, LE);
        let mut chunk = 1u128;
        for i in 1..=m {
            let di = d(b * m + i);
            acc *= di;
            if i < m {
                let p = &powers[i as usize];
                if alternate && i % 2 == 1 {
                    acc -= p;
                } else {
                    acc += p;
                }
            }
            match chunk.checked_mul(u128::from(di)) {
                Some(c) => chunk = c,
                None => {
                    den = &den * &Number::new_from_u128(chunk, LE);
                    chunk = u128::from(di);
                }
            }
        }
        den = &den * &Number::new_from_u128(chunk, LE);
        let top = &(&powers[m as usize] * &g) >> shift_count(f);
        if alternate && m % 2 == 1 {
            acc -= &top;
        } else {
            acc += &top;
        }
        g = &acc / &den;
    }
    let _ = blocks;
    // The block's error is below `3m (ez + 1) + 2.5` and half of the one above it reaches it.
    let err = 6.0 * (m as f64) * (ez + 1.0) + 8.0;
    (g, err)
}

/// The number of terms of a series `Σ c_j z^j` with `|c_j| <= 1/(j+1)!` and `z <= 2^-s` whose tail
/// past them is below `2^-f`: the least `t` with `t s + log2((t + 1)!) >= f + 1`.
pub(super) fn terms(s: u64, f: u64) -> u64 {
    let mut t = 1u64;
    let mut bits = s as f64 + 1.0;
    while bits < f as f64 + 1.0 {
        t += 1;
        bits += s as f64 + log2_below((t + 1) as f64);
    }
    t
}

/// The bits by which an argument is reduced before a series at a precision of `w` bits: about
/// `w^(1/3)`, which balances the series' products against the doublings after it.
pub(super) fn reduction(w: u64) -> u64 {
    (w as f64).cbrt().clamp(4.0, 96.0) as u64
}

/// `e^x` for `x = X 2^-fx > 0` within `ex` units of `2^-fx`, to a relative error below
/// `err 2^-w`, as a floating number of `w` bits and `err`; reduced by `ln 2` first up to the
/// precision of the table [`LN2_LIMBS`].
///
/// `x = 2^s r` with `r < 2^-s0`; `e^r - 1 = r Σ r^j / (j + 1)!` by [`series`], then `s` doublings
/// `e^(2y) - 1 = (e^y - 1)(e^y - 1 + 2)`, in floating numbers, each of which multiplies the
/// relative error by `1 + (e^y - 1)/(e^y + 1) < 2` and adds two units; then 1 is added.
pub(super) fn exp(x: &Number, fx: u64, ex: f64, w: u64) -> (Float, f64) {
    debug_assert!(!x.is_negative());
    if fx > PI_BITS || (x >> shift_count(fx)).bits() < 2 {
        return exp_reduced(x, fx, ex, w);
    }
    // `x = q ln 2 + r` with `0 <= r < ln 2`, `ln 2` from the table within a unit of `2^-fx`: `r`
    // within `ex + q` units, and `e^x = 2^q e^r` with the doublings of `x`'s integer bits saved.
    let ln2 = of_limbs(&LN2_LIMBS) >> shift_count(PI_BITS - fx);
    let xf = Float::of_fixed(x.clone(), fx).to_f64();
    let mut q = (xf / core::f64::consts::LN_2).floor().max(0.0) as u64;
    let mut r = x - &(&ln2 * &Number::new_from_u64(q, LE));
    while r.is_negative() {
        q -= 1;
        r += &ln2;
    }
    while r >= ln2 {
        q += 1;
        r -= &ln2;
    }
    let (mut v, rel) = exp_reduced(&r, fx, ex + q as f64 + 1.0, w);
    v.e += q as i64;
    (v, rel)
}

/// [`exp`] without the reduction by `ln 2`.
fn exp_reduced(x: &Number, fx: u64, ex: f64, w: u64) -> (Float, f64) {
    debug_assert!(!x.is_negative());
    let int_bits = (x >> shift_count(fx)).bits();
    let s0 = reduction(w);
    let s = int_bits + s0;
    // The working precision: the doublings lose up to `s` bits, the small `e^r - 1` `s0`.
    let f = w + s + s0 + 24;
    let r = shift(x, f as i64 - fx as i64 - s as i64);
    // `r` within `ex 2^(f - fx - s) + 1` units of `2^-f`.
    let er = ex * pow2(f as i64 - fx as i64 - s as i64) + 1.0;
    let t = terms(s0, f);
    let (sum, es) = series(&r, f, er, t, |j| j + 1, false);
    // `e^r - 1 = r S`, relatively within `er / r + es / S + 1 / (r S)` (in units of `2^-f`, `r`
    // and `S` integers, `S >= 2^f`).
    let mut y = Float::of_fixed(&r * &sum, 2 * f).cut(f);
    // An integer `X` of `b` bits is at least `2^(b-1)`: `e / X <= e 2^(1-b)`, in units of `2^-f`
    // `e 2^(f+1-b)`. The tail of the series past `t` terms adds a unit to `es`.
    let rel_of = |e: f64, x: &Number| e * pow2(f as i64 + 1 - x.bits() as i64);
    let mut rel = rel_of(er, &r) + rel_of(es + 1.0, &sum) + 2.0;
    for _ in 0..s {
        // `y (y + 2) = y^2 + 2y`: a square, within `2 rel + 2`, and `2y`, within `rel`, of which
        // the sum is within `rel (1 + y / (y + 2)) + 2`, and its cut.
        let ratio = ratio_to_two(&y);
        let square = y.mul(&y, f);
        let double = Float {
            m: y.m.clone(),
            e: y.e + 1,
        };
        y = square.add(&double, f);
        rel = rel * (1.0 + ratio) + 5.0;
    }
    // `e^x = 1 + y`.
    let one = Float {
        m: Number::new_from_u8(1, LE),
        e: 0,
    };
    let v = y.add(&one, f).cut(w);
    // The error of `x` moves `e^x` by `e^(ex 2^-fx) - 1` relatively.
    let dx = ex * pow2(w as i64 - fx as i64) * 1.01;
    let rel = rel * pow2(w as i64 - f as i64) + 2.0 + dx;
    (v, rel)
}

/// `y / (y + 2)` for a positive floating `y`, rounded up, at most 1.
fn ratio_to_two(y: &Float) -> f64 {
    let l = y.log2();
    if l > 60.0 {
        1.0
    } else {
        let v = l.exp2();
        (v / (v + 2.0) * 1.000_001).min(1.0)
    }
}

/// `2^e` as a double, 0 or infinity when out of range.
pub(super) fn pow2(e: i64) -> f64 {
    if e > 1023 {
        f64::INFINITY
    } else if e >= -1022 {
        // A normal number: its exponent field, exactly.
        f64::from_bits(((1023 + e) as u64) << 52)
    } else if e >= -1074 {
        // A subnormal one: its bit in the mantissa.
        f64::from_bits(1u64 << (e + 1074))
    } else {
        0.0
    }
}

/// An upper bound of `log2 x` for `x > 0`, without the system's `log2`: `x = 2^e (1 + m)` with
/// `0 <= m < 1` and `log2(1 + m) - m <= 0.0861` (its largest value, at `m = 1/ln 2 - 1`), so within
/// `0.087` of it.
#[inline]
pub(super) fn log2_above(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let b = x.to_bits();
    let e = (b >> 52) as i64 - 1023;
    let m = f64::from_bits((b & ((1 << 52) - 1)) | (1023 << 52)) - 1.0;
    e as f64 + m + 0.0861
}

/// A lower bound of `log2 x` for `x >= 1`, without the system's `log2`: `x = 2^e (1 + m)` with
/// `0 <= m < 1` and `log2(1 + m) >= m`, the chord of a concave function. Within `0.086` of it; the
/// counts of terms that add it up are then at least what they need.
#[inline]
pub(super) fn log2_below(x: f64) -> f64 {
    debug_assert!(x >= 1.0);
    let b = x.to_bits();
    let e = (b >> 52) as i64 - 1023;
    let m = f64::from_bits((b & ((1 << 52) - 1)) | (1023 << 52)) - 1.0;
    e as f64 + m
}

/// `cos(π a / b)` for `0 <= a < 2b`, to `w` bits, with `pi_f` the fixed-point `π` of `fp` bits
/// (within 4 units): a floating number, its sign, and its relative error in units of `2^-w`;
/// `None` for zero.
///
/// The angle is brought to `π t`, `0 <= t <= 1/2`, as in double precision; then `cos(y) = 1 - v`
/// for `y = π t <= π/4`, and `sin(y) = √(v (2 - v))` for `y = π (1/2 - t)` otherwise, with
/// `v = 1 - cos(y)` from `y 2^-s` by [`series`], `1 - cos(z) = (z^2 / 2) Σ (-z^2)^j 2 / (2j + 2)!`,
/// and `s` doublings `1 - cos(2z) = 2v (2 - v)`, which multiply its relative error by at most
/// `1 + v/(2 - v) < 1.2` and add two units.
pub(super) fn cos_pi(a: u64, b: u64, w: u64, pi_f: &Number, fp: u64) -> Option<(Float, bool, f64)> {
    let (a, b) = (u128::from(a), u128::from(b));
    let a = if a > b { 2 * b - a } else { a };
    let (a, negative) = if 2 * a > b { (b - a, true) } else { (a, false) };
    if 2 * a == b {
        return None;
    }
    let (num, den, sine) = if 4 * a <= b {
        (a, b, false)
    } else {
        (b - 2 * a, 2 * b, true)
    };
    if num == 0 {
        // `cos(0) = 1`.
        return Some((
            Float {
                m: Number::new_from_u8(1, LE),
                e: 0,
            },
            negative,
            0.0,
        ));
    }
    let s = reduction(w);
    // A small angle loses `log2(den)` bits relatively, and `v`, about its square, twice that and
    // twice `s` in fixed point.
    let den_bits = u64::from(128 - den.leading_zeros());
    let f = w + 2 * s + 2 * den_bits + 24;
    debug_assert!(fp >= f);
    // `y 2^-s = π num / den / 2^s`, within `(4 num + 1) / den 2^(fp - f)... + 1` units.
    let y = &(&shift(pi_f, f as i64 - fp as i64 - s as i64) * &Number::new_from_u128(num, LE))
        / &Number::new_from_u128(den, LE);
    let ey = 4.0 * (num as f64) / (den as f64) + 2.0;
    let z = &(&y * &y) >> shift_count(f);
    let ez = 2.0 * ey + 1.0;
    let t = terms(2 * s, f);
    let (sum, es) = series(&z, f, ez, t, |j| (2 * j + 1) * (2 * j + 2), true);
    // `v = z S / 2`.
    let mut v = Float::of_fixed(&z * &sum, 2 * f + 1).cut(f);
    let rel_of = |e: f64, x: &Number| e * pow2(f as i64 + 1 - x.bits() as i64);
    let mut rel = rel_of(ez, &z) + rel_of(es + 1.0, &sum) + 2.0;
    let two = Float {
        m: Number::new_from_u8(2, LE),
        e: 0,
    };
    for _ in 0..s {
        // `2v (2 - v)`, `v < 1`.
        let d = sub(&two, &v, f);
        v = v.mul(&d, f);
        v.e += 1;
        rel = rel * 1.2 + 5.0;
    }
    let (r, rel) = if sine {
        let d = sub(&two, &v, f);
        let p = v.mul(&d, f);
        (sqrt(&p, f), rel * 1.2 / 2.0 + 4.0)
    } else {
        let one = Float {
            m: Number::new_from_u8(1, LE),
            e: 0,
        };
        // `1 - v >= 0.7`: `v`'s error times `v / (1 - v) < 0.42`.
        (sub(&one, &v, f), rel * 0.43 + 2.0)
    };
    Some((r.cut(w), negative, rel * pow2(w as i64 - f as i64) + 2.0))
}

/// Whether `cos(π p / q)` for a reduced `p / q` has a closed form or a chain of the half-angle
/// and triple-angle formulas down to one ([`cos_fixed`]): `q = 2^s 3^t q'` with `q'` 1 or 5.
fn algebraic(q: u64) -> bool {
    let mut q = q >> q.trailing_zeros();
    while q.is_multiple_of(3) {
        q /= 3;
    }
    matches!(q, 1 | 5)
}

/// `cos(π a / b)` for `0 <= a < 2b` as [`cos_pi`] gives it, when the reduced denominator is
/// `2^s 3^t q` with `q` 1 or 5, and `None` otherwise: from the closed forms of the angles `π p / q`
/// for `q` up to 6 (`1/2`, `√2/2`, `√3/2`, `(√5 ± 1)/4`), the half-angle formula for an even
/// denominator, as Arb's `_arb_cos_pi_fmpq_algebraic` takes it, and the triple-angle one for an odd
/// multiple of 3, a root of a cubic ([`cos_fixed`]).
pub(super) fn cos_pi_algebraic(a: u64, b: u64, w: u64) -> Option<Option<(Float, bool, f64)>> {
    let (a, b) = (u128::from(a), u128::from(b));
    let a = if a > b { 2 * b - a } else { a };
    let (a, negative) = if 2 * a > b { (b - a, true) } else { (a, false) };
    if 2 * a == b {
        return Some(None);
    }
    if a == 0 {
        return Some(Some((
            Float {
                m: Number::new_from_u8(1, LE),
                e: 0,
            },
            negative,
            0.0,
        )));
    }
    let g = gcd(a, b);
    let (p, q) = ((a / g) as u64, (b / g) as u64);
    if !algebraic(q) {
        return None;
    }
    let extra = 2 * u64::from(64 - q.leading_zeros()) + 4;
    // The result is at least `sin(π/q) > 2^-bits(q)`: `bits(q)` more for its relative precision.
    let top = w + extra + 16;
    let (c, e) = cos_fixed(p, q, top, extra);
    debug_assert!(!c.is_negative());
    // Relative within `e 2^-top / c` and the cut.
    let x = Float::of_fixed(c, top);
    let low = x.to_f64() * 0.999;
    let rel = e * pow2(w as i64 - top as i64) / low + 2.0;
    Some(Some((x.cut(w), negative, rel)))
}

/// `cos(π a / b)` for `0 <= a < 2b` in fixed point with `f` bits, its magnitude, its sign and a bound
/// of its absolute error in units of `2^-f`, as [`cos_pi_algebraic`] computes it, when the reduced
/// denominator allows (`None` otherwise, and `Some(None)` for zero); `cubic` whether a step of the
/// triple-angle formula may be taken.
pub(super) fn cos_pi_fixed(
    a: u64,
    b: u64,
    f: u64,
    cubic: bool,
) -> Option<Option<(Number, bool, f64)>> {
    let (a, b) = (u128::from(a), u128::from(b));
    let a = if a > b { 2 * b - a } else { a };
    let (a, negative) = if 2 * a > b { (b - a, true) } else { (a, false) };
    if 2 * a == b {
        return Some(None);
    }
    if a == 0 {
        return Some(Some((
            Number::new_from_u8(1, LE) << shift_count(f),
            negative,
            0.0,
        )));
    }
    let g = gcd(a, b);
    let (p, q) = ((a / g) as u64, (b / g) as u64);
    if !algebraic(q)
        || (!cubic && (q >> q.trailing_zeros()).is_multiple_of(3) && q >> q.trailing_zeros() > 3)
    {
        return None;
    }
    let extra = 2 * u64::from(64 - q.leading_zeros()) + 4;
    let (c, e) = cos_fixed(p, q, f + 8, extra);
    // Cut to `f` bits: a unit more.
    Some(Some((&c >> 8usize, negative, e / 256.0 + 1.0)))
}

/// `cos(π p / q)` in fixed point with `f` bits, signed, for `0 <= p <= q` with [`algebraic`]`(q)`
/// (its reduced form), and a bound of its absolute error in units of `2^-f`; `extra` bits more for each
/// level below:
///
/// - `p/q > 1/2`: `-cos(π (q - p) / q)`;
/// - the closed forms of `q <= 6`;
/// - an even `q`: for `p/q <= 1/2`, `√((1 + cos(πp/(q/2)))/2)` when `4p <= q`, and
///   `√((1 - cos(π(q/2 - p)/(q/2)))/2)` otherwise; a root `c = √u` of a `u` within `e` moves by at
///   most `e / (2c)`, `c` at least half its double, which is within `12 u` of it;
/// - an odd multiple of 3: the root `c` near its double of `4c^3 - 3c = t`, `t = cos(3πp/q)`, by
///   Newton's method from the double, the precision doubled at each step, less the bits that the
///   conditioning `f'(c) = 12c^2 - 3` costs, which vanishes at `θ = π/3` ([`cubic`]); the error from
///   the residual `r = 4c^3 - 3c - t`, within `9` units and that of `t`: `|c - c*| <= (|r| + 9 +
///   e_t) / (|f'(c)| / 2)`, as `|f''| <= 28` keeps `|f'|` above half its value within `|f'| / 64` of
///   `c`, a distance the bound and the double's are checked to be below, which keeps the other roots
///   of the cubic, `cos(θ ± 2π/3)`, out.
fn cos_fixed(p: u64, q: u64, f: u64, extra: u64) -> (Number, f64) {
    let g = gcd(u128::from(p), u128::from(q)) as u64;
    let (p, q) = (p / g, q / g);
    if 2 * p > q {
        let (c, e) = cos_fixed(q - p, q, f, extra);
        return (-c, e);
    }
    let one = Number::new_from_u8(1, LE) << shift_count(f);
    if 2 * p == q {
        return (Number::new_from_u8(0, LE), 0.0);
    }
    if p == 0 {
        return (one, 0.0);
    }
    // `isqrt(x 2^(2f))`, `√x` rounded down, within a unit.
    let root_of = |x: u64| (Number::new_from_u64(x, LE) << shift_count(2 * f)).isqrt();
    match (p, q) {
        // `cos(π/3) = 1/2`, exactly.
        (1, 3) => (&one >> 1usize, 0.0),
        // `√2/2 = √(1/2)` and `√3/2 = √(3/4)`: a unit, after the exact shift of the radicand.
        (1, 4) => (
            (Number::new_from_u8(1, LE) << shift_count(2 * f - 1)).isqrt(),
            1.0,
        ),
        (1, 6) => (
            (Number::new_from_u8(3, LE) << shift_count(2 * f - 2)).isqrt(),
            1.0,
        ),
        // `(1 ± √5)/4`: the root within a unit, a quarter of it after the shift, which rounds down
        // by less than a unit.
        (1, 5) => (&(&one + &root_of(5)) >> 2usize, 2.0),
        (2, 5) => (&(&root_of(5) - &one) >> 2usize, 2.0),
        _ if q.is_multiple_of(2) => {
            let h = q / 2;
            let (inner, plus) = if 4 * p <= q {
                (p, true)
            } else {
                (h - p, false)
            };
            // `cos(π inner / h)`, `inner / h <= 1/2`, at `extra` bits more.
            let (c, ec) = cos_fixed(inner, h, f + extra, extra);
            let onex = Number::new_from_u8(1, LE) << shift_count(f + extra);
            let u = if plus { &onex + &c } else { &onex - &c };
            // `u / 2` within `ec / 2` units of `2^-(f + extra)`, and its root in fixed point with `f`
            // bits: `√(U 2^-(f+extra) / 2) 2^f = √(U 2^(f - extra - 1))`.
            let shifted = if f > extra {
                &u << shift_count(f - extra - 1)
            } else {
                &u >> shift_count(extra + 1 - f)
            };
            let r = shifted.isqrt();
            // The root moves by `(ec/2) 2^-(f+extra) / (2c)` at most, `c` the result, at least half
            // its double; in units of `2^-f`, and one for the rounding.
            let c_low = super::double::cos_pi(p, q) * 0.5;
            let e = ec / 2.0 * pow2(-(extra as i64)) / (2.0 * c_low) + 1.0;
            (r, e)
        }
        _ => {
            debug_assert!(q.is_multiple_of(3) && q > 6);
            cubic(p, q, f, extra)
        }
    }
}

/// `cos(π p / q)` for `0 < p/q < 1/2` and an odd multiple `q` of 3, by the triple-angle formula
/// (see [`cos_fixed`]), and the bound of its error in units of `2^-f`.
fn cubic(p: u64, q: u64, f: u64, extra: u64) -> (Number, f64) {
    // `t = cos(3πp/q) = cos(π p / (q/3))`, `p / (q/3) < 3/2`: reduced to `[0, 1]` by
    // `cos(π x) = cos(π (2 - x))`.
    let h = q / 3;
    let inner = if p > h { 2 * h - p } else { p };
    let (t, et) = cos_fixed(inner, h, f + extra, extra);
    let start = super::double::cos_pi(p, q);
    // The conditioning: `f'(c) = 12c^2 - 3`, about `d0` at the root, vanishes at `θ = π/3`, where
    // `4c^3 - 3c = -1` has a double root, and the angles of a denominator `3^j` next to `π/3` come
    // within `π / 3^j` of it. A step's residual, within about 10 units, moves its correction by
    // `10 / d0` units, the `guard` bits each precision carries; its quadratic constant
    // `f'' / (2f') = 12c / f'(c)`, at most `12 / d0`, costs `lost` bits a step.
    // Kept above `10^-12`, which no reduced denominator below `2^35` comes near: its angles are at
    // least `π/q > 10^-10` off `π/3`.
    let d0 = (12.0 * start * start - 3.0).abs().max(1e-12);
    let guard = (16.0 / d0).log2().ceil().max(0.0) as u64;
    let lost = (12.0 / d0).log2().ceil().max(0.0) as u64 + 1;
    // The bits each step makes right, from `f + 16` down: a step from `s` right bits makes about
    // `2s - lost` right at its precision, `guard` bits above them; down to what the double holds,
    // 49 bits (within `12 u` of the root, which is below 1), or `lost + 18`, below which the halving
    // would stop falling. Without the guard and the constant's bits a step near `π/3` fell short by a
    // few bits, a shortfall that doubled at each step: `cos(2188π/6561)` to 4000 bits was right to
    // fewer than 3900, and to 16000 its residual did not fit a double.
    let mut right = vec![f + 16];
    let floor = 49u64.max(lost + 18);
    while *right.last().expect("not empty") > floor {
        let s = *right.last().expect("not empty");
        right.push((s + lost) / 2 + 8);
    }
    // A step of Newton's method at `pr` bits from `c` of `fc` bits: `c - f(c) / f'(c)` with
    // `f(x) = 4x^3 - 3x - t` and `f'(x) = 12 x^2 - 3`.
    let step = |c: &Number, fc: u64, pr: u64| -> Number {
        let x = shift(c, pr as i64 - fc as i64);
        let tp = shift(&t, pr as i64 - (f + extra) as i64);
        let x2 = shift(&(&x * &x), -(pr as i64));
        let x3 = shift(&(&x2 * &x), -(pr as i64));
        let fx = &(&(&x3 << 2usize) - &(&x * 3u32)) - &tp;
        let three = Number::new_from_u8(3, LE) << shift_count(pr);
        let dfx = &(&x2 * 12u32) - &three;
        let correction = &(&fx << shift_count(pr)) / &dfx;
        &x - &correction
    };
    let mut c = Number::new_from_u64((start * pow2(60)) as u64, LE);
    let mut fc = 60u64;
    // The last of `right` is what the start holds.
    for &s in right.iter().rev().skip(1) {
        c = step(&c, fc, s + guard);
        fc = s + guard;
    }
    // The residual at the precision of the last step: `x^2` within a unit, `x^3` within
    // `|x| + 1 <= 2`, `4 x^3` within 8, and `t` within `et 2^-extra`; then `|c - c*| <=
    // (|r| + er) / (|f'| / 2)` units of `2^-pr`, `dist`, while `|f'|` stays above half (below).
    let pr = fc;
    let bound = |c: &Number| -> (f64, f64, f64) {
        let x2 = shift(&(c * c), -(pr as i64));
        let x3 = shift(&(&x2 * c), -(pr as i64));
        let tp = shift(&t, pr as i64 - (f + extra) as i64);
        let r = &(&(&x3 << 2usize) - &(c * 3u32)) - &tp;
        let xf = Float::of_fixed(c.clone(), pr).to_f64();
        let dfx = (12.0 * xf * xf - 3.0).abs();
        let er = 8.0 + 1.0 + et * pow2(pr as i64 - (f + extra) as i64);
        let dist = (Float::of_fixed(r, 0).to_f64().abs() + er) / (dfx / 2.0) * 1.001;
        (dist, dfx, xf)
    };
    let (mut dist, mut dfx, mut xf) = bound(&c);
    // Steps more at the last precision while `c` is short of `f + 16` right bits by more than 10: a
    // guard for a conditioning that the double misjudged.
    for _ in 0..4 {
        if dist.is_finite() && dist.log2() <= (guard + 10) as f64 {
            break;
        }
        c = step(&c, pr, pr);
        (dist, dfx, xf) = bound(&c);
    }
    // Within `δ = |f'(c)| / 64` of `c`, `|f''| = 24 |y| <= 28` keeps `|f'| >= |f'(c)| / 2`: `f` is
    // monotone there, so it has one root within `dist` of `c` when that is below `δ`, and no other.
    // `cos(πp/q)` is a root within `|c - start| + 12 u` of `c`, the double being within `12 u` of it,
    // so it is that root when that is below `δ` too. (Near `θ = π/3` the other roots,
    // `cos(θ ± 2π/3)`, come within `√3 |sin(θ - π/3)|` of it, which the check keeps out of `δ`.) In
    // logarithms: `2^-pr` is no double past 1074 bits.
    let delta = dfx / 64.0;
    let away = (xf - start).abs();
    assert!(
        dist.is_finite()
            && dist.log2() - (pr as f64) < delta.log2()
            && away + 16.0 * super::double::U < delta,
        "Newton's method on the triple angle did not converge"
    );
    // Rounded down to `f` bits: a unit more, the distance in units of `2^-f`.
    (
        shift(&c, f as i64 - pr as i64),
        dist * pow2(f as i64 - pr as i64) + 1.0,
    )
}

/// `a - b` for floating numbers with `a > b > 0`, cut to `w` bits.
fn sub(a: &Float, b: &Float, w: u64) -> Float {
    let e = a.e.min(b.e);
    Float {
        m: &shift(&a.m, a.e - e) - &shift(&b.m, b.e - e),
        e,
    }
    .cut(w)
}

/// `√x` for a positive floating number, to `w` bits, rounded down: a unit lost, and half the
/// relative error of `x`.
pub(super) fn sqrt(x: &Float, w: u64) -> Float {
    // An even exponent, and `2w + 2` bits under the root.
    let bits = x.m.bits() as i64;
    let mut k = 2 * w as i64 + 2 - bits;
    if (x.e - k) % 2 != 0 {
        k += 1;
    }
    let m = shift(&x.m, k).isqrt();
    Float {
        m,
        e: (x.e - k) / 2,
    }
    .cut(w)
}

/// `x^k` for `k >= 1`, cut to `w` bits after each product, from the top bit of `k`, and the number
/// of products, each of which adds two units of `2^-w` to the relative error.
fn power(x: &Float, k: u64, w: u64) -> (Float, u64) {
    let mut r = x.clone().cut(w);
    let mut products = 0;
    let bits = 64 - k.leading_zeros();
    for i in (0..bits - 1).rev() {
        r = r.mul(&r, w);
        products += 1;
        if (k >> i) & 1 == 1 {
            r = r.mul(x, w);
            products += 1;
        }
    }
    (r, products)
}

/// `1 - a x^k` in fixed point with `f` bits, for floating `a` and `x` with `a x^k` near 1, the
/// power and the product cut to `p` bits, and the number of products.
fn residual(a: &Float, x: &Float, k: u64, p: u64, f: u64) -> (Number, u64) {
    let (v, products) = power(x, k, p);
    let av = a.clone().cut(p).mul(&v, p);
    let one = Number::new_from_u8(1, LE) << shift_count(f);
    (&one - &av.to_fixed(f), products + 1)
}

/// `a^(-1/k)` for a positive floating `a` within `ra` units of `2^-wa` relatively, `k >= 2`, to `w`
/// bits, and its relative error in units of `2^-w`.
///
/// Newton's iteration for `a^(-1/k)`, which takes no division: `x + x (1 - a x^k) / k`, the
/// precision doubled at each step from a double's start, less the bits of the step's quadratic
/// constant: for `x = a^(-1/k) (1 + δ)` the step gives `a^(-1/k) (1 - (k + 1) δ^2 / 2 + ...)`, so
/// from `s` right bits about `2s - log2((k + 1) / 2)`. Its error is bounded afterwards from the
/// residual `t = 1 - a x^k` computed once more: with `x = a^(-1/k) (1 + δ)`, `(1 + δ)^k = 1 - t`,
/// so `|δ| <= |t| / (k (1 - |t|))`; `t` is within a unit of the fixed point and two for each cut of
/// the power and the product, and `a`'s own error reaches `x` divided by `k`.
pub(super) fn inverse_root(a: &Float, ra: f64, wa: u64, k: u64, w: u64) -> (Float, f64) {
    // The start: `2^(-log2(a) / k)`, right to about `50 - log2 |l|` bits (the roundings of the
    // double `l` and of `log2 a`, divided by `k`).
    let l = -a.log2() / k as f64;
    let e0 = l.floor();
    let mant = ((l - e0).exp2() * pow2(52)) as u64;
    let mut x = Float {
        m: Number::new_from_u64(mant, LE),
        e: e0 as i64 - 52,
    };
    // The precisions from the target down: each step reaches its precision from half of it plus
    // the constant's bits, `lost >= log2((k + 1) / 2) + 1`, and 8 to spare, down to what the start
    // already holds; at least `lost + 18`, below which the halving would stop falling. Without
    // `lost`, a step fell short by about that many bits, a shortfall that doubled at each step: at
    // `k = 643` and 4458 bits `x` was right to fewer than 4300, and `p(10^11)` failed its bound.
    let lost = u64::from(64 - k.leading_zeros());
    let start = ((50.0 - (l.abs() + 1.0).log2()).max(0.0) as u64).max(lost + 18);
    let target = w + 16;
    let mut precisions = vec![target];
    while *precisions.last().expect("not empty") > start {
        let p = *precisions.last().expect("not empty");
        precisions.push((p + lost) / 2 + 8);
    }
    for &p in precisions.iter().rev().skip(1) {
        // Room for the correction, of about half the bits.
        let mut xm = x.clone();
        let have = xm.m.bits();
        if have < p {
            xm.m = &xm.m << shift_count(p - have);
            xm.e -= (p - have) as i64;
        }
        let f = p + 8;
        let (t, _) = residual(a, &xm, k, p + 8, f);
        // `x (1 + t / k)`: the correction `x t / k` at `x`'s exponent.
        let d = &(&(&xm.m * &t) >> shift_count(f)) / k;
        x = Float {
            m: &xm.m + &d,
            e: xm.e,
        }
        .cut(p);
    }
    let p = target + 8;
    let (mut t, mut products) = residual(a, &x, k, p, p);
    // Steps more at the target while the residual shows `x` short of it (`|t| / k` above about
    // `2^-target`): a guard for a start the schedule misjudged, each step at least doubling the
    // right bits.
    let enough = lost + 12;
    for _ in 0..4 {
        if t.bits() <= enough {
            break;
        }
        let mut xm = x.clone();
        let have = xm.m.bits();
        if have < target {
            xm.m = &xm.m << shift_count(target - have);
            xm.e -= (target - have) as i64;
        }
        let d = &(&(&xm.m * &t) >> shift_count(p)) / k;
        x = Float {
            m: &xm.m + &d,
            e: xm.e,
        }
        .cut(target);
        (t, products) = residual(a, &x, k, p, p);
    }
    // `|t|` and its error, in units of `2^-w`: far below 1 as fractions, so `1 - |t| > 0.999`.
    let tu = t.to_f64().abs() * pow2(w as i64 - p as i64);
    let eu = (1.0 + 2.0 * products as f64 + ra * pow2(p as i64 - wa as i64) + 2.0)
        * pow2(w as i64 - p as i64);
    let bound = (tu + eu) / (k as f64 * 0.999);
    let rel = bound + ra * pow2(w as i64 - wa as i64) / k as f64 + 2.0;
    (x.cut(w), rel * 1.001)
}

/// `(T_m(x), U_(m-1)(x))`, the Chebyshev polynomials of `x = X 2^-f` (`|x| <= 1`, exact), in fixed
/// point with `f` bits, and bounds of their absolute errors in units of `2^-f`: from `(T_1, U_0) =
/// (x, 1)` by the bits of `m`, `(T_(2j), U_(2j-1)) = (2 T_j^2 - 1, 2 T_j U_(j-1))` and
/// `(T_(j+1), U_j) = (x T_j - (1 - x^2) U_(j-1), x U_(j-1) + T_j)`, `|T_j| <= 1` and
/// `|U_(j-1)| <= j`.
fn chebyshev(x: &Number, f: u64, m: u64) -> (Number, Number, f64, f64) {
    let one = Number::new_from_u8(1, LE) << shift_count(f);
    let mut t = x.clone();
    let mut u = one.clone();
    let (mut et, mut eu) = (0.0f64, 0.0f64);
    let x2 = &(x * x) >> shift_count(f);
    let one_minus = &one - &x2;
    let mut j = 1u64;
    for i in (0..63 - m.leading_zeros()).rev() {
        let tt = &(&t * &t) >> shift_count(f);
        let tu = &(&t * &u) >> shift_count(f);
        t = &(&tt << 1usize) - &one;
        u = tu << 1usize;
        let (et0, eu0) = (et, eu);
        // A product of two errors in units is in units of `2^-2f`.
        et = 2.0 * (2.0 * et0 + et0 * et0 * pow2(-(f as i64)) + 1.0);
        eu = 2.0 * (eu0 + j as f64 * et0 + et0 * eu0 * pow2(-(f as i64)) + 1.0);
        j *= 2;
        if (m >> i) & 1 == 1 {
            let xt = &(x * &t) >> shift_count(f);
            let wu = &(&one_minus * &u) >> shift_count(f);
            let xu = &(x * &u) >> shift_count(f);
            let (et0, eu0) = (et, eu);
            u = &xu + &t;
            t = &xt - &wu;
            et = et0 + 1.0 + (j as f64 + eu0 + 1.0) + 1.0;
            eu = eu0 + 1.0 + et0;
            j += 1;
        }
    }
    (t, u, et, eu)
}

/// The largest reduced denominator for which [`cos_pi_newton`] is used: past them, the bound of
/// the polynomial's second derivative no longer keeps its first above half its value at the root
/// over the start's distance.
pub(super) const NEWTON_DENOMINATOR: u64 = 512;

/// `cos(π a / b)` for `0 <= a < 2b` as [`cos_pi`] gives it, by Newton's method on a Chebyshev
/// polynomial with a simple root there, for a reduced angle `π a' / b'` with `b' <=`
/// [`NEWTON_DENOMINATOR`]: `T_(b'/2)` for an even `b'` (`a'` is odd and `T_(b'/2)(cos(π a'/b')) =
/// cos(π a'/2) = 0`), and `U_(b'-1)` for an odd one (`sin(π a') = 0`), whose derivative is
/// `(b' T_b' - x U_(b'-1)) / (x^2 - 1)`. The precision is doubled at each step from the double's
/// start.
///
/// The error is bounded afterwards: with `f(x~)` within `E`, `|x~ - x| <= (|f(x~)| + E) / (m/2)`,
/// as `|f'| = m / sin(θ)` or `m / sin(θ)^2` at the root is at least `m` (the polynomial's degree),
/// and by Markov's inequality `|f''| <= m^5 / 3` keeps `|f'| >= m/2` within `1.5 / m^4 >= 2^-36` of
/// the root, a distance the result is checked to be within.
pub(super) fn cos_pi_newton(a: u64, b: u64, w: u64) -> Result<Option<(Float, bool, f64)>, ()> {
    let (a, b) = (u128::from(a), u128::from(b));
    let a = if a > b { 2 * b - a } else { a };
    let (a, negative) = if 2 * a > b { (b - a, true) } else { (a, false) };
    if 2 * a == b {
        return Ok(None);
    }
    if a == 0 {
        return Ok(Some((
            Float {
                m: Number::new_from_u8(1, LE),
                e: 0,
            },
            negative,
            0.0,
        )));
    }
    let g = gcd(a, b);
    let (a, b) = ((a / g) as u64, (b / g) as u64);
    if b > NEWTON_DENOMINATOR {
        return Err(());
    }
    let (m, even) = if b % 2 == 0 {
        (b / 2, true)
    } else {
        (b, false)
    };
    let bits = u64::from(64 - b.leading_zeros());
    let target = w + 3 * bits + 24;
    let x0 = super::double::cos_pi(a, b);
    let start = Number::new_from_u64((x0 * pow2(60)) as u64, LE);
    let mut x = start.clone();
    let mut fx = 60u64;
    // A step from `s` right bits makes `2s - log2 c` right for the quadratic constant `c = |f''| /
    // (2 |f'|)` at the root, `cos θ / (2 sin^2 θ)` for `T_m` and three times that for `U_(m-1)`, below
    // `2^(2 bits(b))`: each precision is reached from half of it plus those bits. Without them, at
    // `cos(π/511)` the steps fell short by about two bits, a shortfall that doubled at each step: to
    // 16,000 bits its bound was `4 10^18` units.
    let lost = 2 * bits;
    let mut precisions = vec![target];
    while *precisions.last().expect("not empty") > 90 {
        let p = *precisions.last().expect("not empty");
        precisions.push((p + lost) / 2 + 8);
    }
    for &p in precisions.iter().rev() {
        let xp = shift(&x, p as i64 - fx as i64);
        x = newton_step(&xp, p, m, even);
        fx = p;
    }
    // The distance to the start, within `2^-41`, keeps the root within `2^-40` (the double within
    // `12 u` of it).
    let d = &shift(&start, fx as i64 - 60) - &x;
    if d.bits() + 41 > fx {
        return Err(());
    }
    let (t, u, et, eu) = chebyshev(&x, fx, m);
    let (value, e) = if even { (t, et) } else { (u, eu) };
    // `|x~ - x| <= (|f| + E) 2 / m`, in units of `2^-fx`; then relatively, `x >= x~ - that`.
    let dist = (value.to_f64().abs() + e + 1.0) * 2.0 / m as f64;
    let x = Float::of_fixed(x, fx);
    let low = x.to_f64() * (1.0 - 2.0 * super::double::U) - dist * pow2(-(fx as i64));
    let rel = dist * pow2(w as i64 - fx as i64) / low * 1.001 + 2.0;
    Ok(Some((x.cut(w), negative, rel)))
}

/// One step of Newton's method for the root of `T_m` (`even`) or `U_(m-1)` at `x = X 2^-p`.
fn newton_step(x: &Number, p: u64, m: u64, even: bool) -> Number {
    let (t, u, _, _) = chebyshev(x, p, m);
    let correction = if even {
        // `T_m / (m U_(m-1))`.
        &(&t << shift_count(p)) / &(&u * m)
    } else {
        // `U_(m-1) (x^2 - 1) / (m T_m - x U_(m-1))`.
        let one = Number::new_from_u8(1, LE) << shift_count(p);
        let x2m1 = &(&(x * x) >> shift_count(p)) - &one;
        let num = &u * &x2m1;
        let den = &(&t * m) - &(&(x * &u) >> shift_count(p));
        &num / &den
    };
    x - &correction
}

/// The greatest common divisor.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
