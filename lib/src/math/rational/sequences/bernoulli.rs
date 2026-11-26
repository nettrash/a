//! The Bernoulli numbers `B_n`, with `B_1 = -1/2`.
//!
//! Up to `B_200` from tables. Beyond, `|B_n| = 2 n! ζ(n) / (2π)^n` ([`by_zeta`]): the denominator
//! `d` by von Staudt and Clausen's theorem, `K = 2 d n!` exactly, and `K ζ(n) / (2π)^n` to the bits
//! of `|B_n| d` and 8 more, rounded; `π` from the table of the partition numbers (16,384 bits) or
//! their Chudnovsky series, `(2π)^n` by powers cut to the precision, `1/ζ(n)` by its Euler product
//! over the primes `p` with `p^n` below `2^w`, each factor `z <- z - z / p^n` taken to the bits it
//! moves the product by. From `n = 8000` the low bits of `|B_n| d` come from its residues modulo
//! primes ([`modp`](super::modp)), joined by the Chinese remainder theorem, and the formula needs
//! only the rest: the multiple of their product nearest to the formula's value is added.
//!
//! The products and the quotients of the formula reuse a few vectors ([`Work`]): the squares and
//! products by the numbers' kernels into them, the quotients by divisors of up to
//! [`SCHOOL_DIVISION`] limbs by Knuth's algorithm D in them, so that a short number allocates a
//! handful of times rather than at every step.
//!
//! The table of all the numbers up to an index ([`bernoulli_numbers`]) is read from the tables up to
//! `B_200`, and beyond computed by the same formula from the top index down ([`table_by_zeta`]).

use super::super::Rational;
use super::limbs::{
    Limbs, bits, mul, mul_1, normalize, number_u128, odd_primes, of_number, product, shl, shl_into,
    shr, shr_assign, shr_into, sub, to_number,
};
use crate::math::numbers::nat::{div_2by1, invert_limb, mul_into, sqr_into};
use crate::math::numbers::{ByteOrder, Number, PI_TABLE, is_prime_u64, pi, primes};

/// `B_0, B_2, ..., B_58` as (numerator, denominator): `B_60` is the first whose numerator passes
/// `2^127`.
pub(super) const TABLE: [(i128, u64); 30] = [
    (1, 1),
    (1, 6),
    (-1, 30),
    (1, 42),
    (-1, 30),
    (5, 66),
    (-691, 2730),
    (7, 6),
    (-3617, 510),
    (43867, 798),
    (-174611, 330),
    (854513, 138),
    (-236364091, 2730),
    (8553103, 6),
    (-23749461029, 870),
    (8615841276005, 14322),
    (-7709321041217, 510),
    (2577687858367, 6),
    (-26315271553053477373, 1919190),
    (2929993913841559, 6),
    (-261082718496449122051, 13530),
    (1520097643918070802691, 1806),
    (-27833269579301024235023, 690),
    (596451111593912163277961, 282),
    (-5609403368997817686249127547, 46410),
    (495057205241079648212477525, 66),
    (-801165718135489957347924991853, 1590),
    (29149963634884862421418123812691, 798),
    (-2479392929313226753685415739663229, 870),
    (84483613348880041862046775994036021, 354),
];

/// The magnitudes of the numerators of `B_60, B_62, ..., B_200`, their limbs one after another,
/// the least significant first; the number of `B_(60 + 2i)` ends at [`LONG_END`]`[i]`. The signs
/// are those of every Bernoulli number: negative for `n = 0 mod 4`. Generated and checked against
/// PARI's `bernfrac` and FLINT's `bernoulli_fmpq_ui` by an exact recurrence (`Σ C(m+1, k) B_k = 0`).
const LONG: [u64; 492] = [
    0x5047DAA8A7E64393,
    0x3FCD77896A343F43,
    0x0000000000000DF3,
    0xD9547361F5E55927,
    0x094101AF5D50D145,
    0x7F11711CAE210B81,
    0xCF399B169AABFC7C,
    0x0000000000000139,
    0x5FEB781D2A70B25D,
    0x847FD77286DFF384,
    0x000000000042089E,
    0xB76765ED9A363C79,
    0x6D0164E132C16CA9,
    0x0000000000038845,
    0x340AD867515A816B,
    0x0F1D5E8B97A3A66F,
    0x0000000107AFA25E,
    0xD17BE20C9315664D,
    0x12515C307E590205,
    0x003CD8C09A3E273F,
    0x22D78D2246E465AF,
    0x0323A2D34FC8423E,
    0x000000175E375B58,
    0xBCFBFEFECA7FFB09,
    0xD6CE058EF0E2F29A,
    0x000041E5B24381A3,
    0x774CC3B79EB45C9F,
    0xF685050B1191EA30,
    0x10EB32B9DC714E8B,
    0xB6FE7C5D41F609A3,
    0x0E0046E172E59E07,
    0x6CD6D086747768A8,
    0x00000000000002DD,
    0xA5075F246809594D,
    0x82F2C644816BFBB7,
    0x29ECAC4470D12BC3,
    0x000000000000010B,
    0xFE6D75CA93E13F3D,
    0x9F3DAC68E63AA1CD,
    0x1F2002A1046FF4C6,
    0x00000000133978DC,
    0x962077BFD79084B7,
    0x8AAE8D75791166F6,
    0xE95D0D91386304FB,
    0x0000000000019B29,
    0x2F2C2AA82F64BADF,
    0xD554D3D23FC9E873,
    0x242495C81AFBF274,
    0x00000030A4BD337B,
    0xB89A85CA61976D67,
    0xD852B7E8124D88BC,
    0x95346A5418CCE492,
    0x0000AAD5AA44E228,
    0x6CD99B8D53E67587,
    0xBE2C3FA060C5595B,
    0xA120545F9D05C713,
    0x0000BBB7F42443C5,
    0xD6EA05C62EBAB1E7,
    0xE74ACAEF0D671282,
    0xE7EC42F88F6ADC1F,
    0x0000B0E2803490F4,
    0xF4FF3683FCBE99BB,
    0x5DD07A328BA43B65,
    0x20D18760F53B1C62,
    0x6ACA47D00BE6048D,
    0x0000000000000723,
    0xED452F06F898C63F,
    0xC25CF05D9FA06D10,
    0x21C31E201EDD6FA9,
    0x9622B6A5B44A3F7A,
    0x2F4D0E7B433468DF,
    0xB5944A5E7E797ACD,
    0xE99995EDA6EBF5A3,
    0x997E8AEAE3D14E99,
    0x00000000000C7744,
    0xF3B5AC7F3FF729C7,
    0x397D5914F12B9CD8,
    0xCDAC622BB1082554,
    0xAF9CFCDFECDBD340,
    0x0000000001A637B3,
    0xB3D7804C77980375,
    0xFB79DA5F5E53B8D9,
    0x0AE699A11A3F3125,
    0x7ABD271ACE9E8A2D,
    0x00000000A47B55BB,
    0x04E868487E944C65,
    0x21A464FD64875B1D,
    0x33CF96DAF35F8288,
    0x68981C301464A64B,
    0x0000004923AD1715,
    0xD60CA7CC5E0B9BC9,
    0xCCADA531F98355E8,
    0x503DB7DA1D455166,
    0x49DAA85DEEC289D3,
    0x9FCDC8928D1DC292,
    0x0000000000000001,
    0x22B77E46C4E5182B,
    0x8059051C4734B5B1,
    0x498DA98A833E512F,
    0xF656ECAC7BFB8A23,
    0x00EA97D492FB2ABD,
    0x605B8B714117002D,
    0xBF72E68B651B4A70,
    0xB52CF65682BE0E00,
    0x12D42690D4B918F0,
    0x0E585391AB427A9B,
    0x00000000000004D9,
    0x2D7159315A505C59,
    0x4C9E506100F87EF8,
    0xB1982767DFB2ED27,
    0xE95A262F9EAE6FCD,
    0x2D4C90F12736D96D,
    0x000000000000000A,
    0x626AA61BF9C58903,
    0x50496B2F15E61FA9,
    0xE27E1E7F1852F3BF,
    0x70AAAC1230710E43,
    0x200B403E406B8CB1,
    0x000000000002361B,
    0x66B747E709459577,
    0x0C800670258A6429,
    0xF797318A49C7E779,
    0xD344A38473B331D9,
    0x41950DBC20F27418,
    0x0000000000029F18,
    0x1097F8923A1FFA73,
    0xA0E798F1F57ECF67,
    0xF6F1D67E2111556A,
    0xC93A14239E876528,
    0x3E33FA9DCA3EB62C,
    0x0055ABAE5049FB29,
    0xDFE022DCBAC570CF,
    0xB9FFDFE00DD71C9A,
    0xDB160A31C7881716,
    0xF685B5E723599A04,
    0xFF479520D1E0EF11,
    0x0000000569069737,
    0x7C6F0ED50C08C869,
    0x31640D0B031C9949,
    0x92093623086615F2,
    0x4735A013929F7C0A,
    0x18E5CC3BBA0B9F05,
    0x000028D2EE1E4628,
    0xCF99F4C9E3AB07BF,
    0x895F3AA893B87FD4,
    0xD795175175DFCA6D,
    0xDFE0D97BAE982EAE,
    0xCCF0B6455B2718A3,
    0x0434CFD60BCFCBCA,
    0x000000000000008D,
    0x5971F67262899801,
    0xC0B5EA74BB8E5EAB,
    0x984C9C89CFDC0712,
    0xF192E34D8D53FB98,
    0xBCF8504B386383F8,
    0xCBA32F410BF8AD4E,
    0x0000000000000006,
    0x6CCAF7075031CD5F,
    0xC689B117DC9806BD,
    0x4A69A8935A4F0D9C,
    0x25EBAD9C0F35C7B8,
    0x0B6E9FDD92910057,
    0xFAC9E66490989C30,
    0x000000000000BF28,
    0x2FEF024ABD4A948F,
    0xABF4CAB7C6CCFDB6,
    0x4AC179A221A72207,
    0x409A807E10381A27,
    0xF7D66D564491F792,
    0xE9E16656322C3D1F,
    0x000000026DA8EB14,
    0xD21D75E2B7E379D7,
    0xBCDD25D28F2CFF67,
    0x56F16DCF6A333719,
    0x52E8A20AF487B367,
    0x2141C728CD8B2699,
    0xAB595D08E95901BF,
    0x0000000000667708,
    0x4F305B92094C6D09,
    0x30B213BA98155679,
    0x3B1CEA321CCDF50C,
    0x2B7E89C44050B595,
    0x4E5111E2A0E8D8D1,
    0x60CAC802DFAAADBA,
    0x000001F21474E760,
    0x92BC0475351CF8ED,
    0x4ABC7CBB4EDE97A0,
    0x0BF439B143B9FD64,
    0xF8E254E47EC5A252,
    0xCC3A88EF03E4FFD0,
    0xEBBE7CAAC06AC603,
    0x00F2FC1CF199FF39,
    0x47F8388A5C1947C1,
    0xB0FC29A2C6CE0D82,
    0x459F2C44C216F2C0,
    0xAC4F4F2849642524,
    0xBF7E996BD12A326E,
    0x4B35A627A5C36531,
    0x86979BDA57E904E0,
    0x0000000000000004,
    0x4A2934E944848307,
    0xE987D72B96EA3A12,
    0x8F3AF8040E8CB359,
    0x64807627AB4EFABE,
    0x8989D9E23E84F98C,
    0x680B9C98DABB2D78,
    0x05304F3742678E34,
    0xD8677A2D17BF06ED,
    0x1F0FF3D59DE06343,
    0xB7B0BDC1543FF458,
    0xEF5510B9656CAF21,
    0x68DC597EB3F31279,
    0x712B5A188BBD7801,
    0xFF93A54FCF1414B9,
    0x00000000FA22E4C8,
    0xE11B4AEBC7B0E55F,
    0xC4A3BA68C1CBC94D,
    0x4969043DBBE6A15B,
    0x1E72A06B25F7E411,
    0xFDF231D24FCB3EBF,
    0x7B7A1E64D1A209E5,
    0x2AB8F4553068D227,
    0x0000000000001625,
    0x53A342D3DBA8F28D,
    0xB66920E4DD23C2FF,
    0xCBE1F5109D314E41,
    0xFBD5C76A592525A5,
    0x6DDC48B0A6DB6639,
    0xC11BB5A2C324F964,
    0xCE0055756D2A72FB,
    0x000000008ABB4070,
    0xDD5F1DE573417BA3,
    0x96C525AD3620948A,
    0x49B6256E82028B3E,
    0xB20848B1E503131E,
    0xCD3DE0499D764BAD,
    0x32CB07B0EBEA931E,
    0x303D12C1DF241D36,
    0x000243CF6C45E5F3,
    0x53187949EDA861A1,
    0xEAA7A2EBD799872B,
    0x3BD92084545458F4,
    0xC872D37F6ED5D765,
    0x1479931CF23DAF9D,
    0x51947B47F3455958,
    0x630567C6050C880B,
    0x000004AD1761C335,
    0xC28CCB998937AFD9,
    0xB994AAEA70C03FEB,
    0xBCFF913D225748DB,
    0x054C31F0CC075375,
    0x2FD3C37981CA9DC0,
    0xB70EA33AF2E98BEE,
    0x9408860E8050E502,
    0x003226017E58D256,
    0xA449D8C5D9A26E5D,
    0x97CD44046183627E,
    0x9CE0FAFD4F764E1E,
    0x6EB264369DC354D7,
    0xB0C9F5D535153418,
    0x431649B5E4754FA1,
    0xCD8CB2CB2B428653,
    0x4C9228F47E9CA983,
    0x00000000005CFFDD,
    0x7641BC0B74984367,
    0x753894C223677A27,
    0x867CAF70C8A63BF5,
    0xB0044E4359358393,
    0x493287850E093CB4,
    0x400EC4D01E91FB6A,
    0xF49A6D4AE379A5EF,
    0xCDCE90B494A77BCC,
    0x000000000000000C,
    0x4AF9D695406AFC83,
    0xC4D7F639B68FC655,
    0xFA6DF268B098C702,
    0xF72EB54C698447F7,
    0x91C6E3C5A9193DC3,
    0xAECDFE66A63546B7,
    0xD00546FE032B7154,
    0xF7B2690499CB998A,
    0x0000000012DA5277,
    0x2668D3836C89BF91,
    0xF2D67B5933C4884E,
    0xEF41B10BBB68600F,
    0xA022AF121FEF8246,
    0x8E102E14B6EAB8C2,
    0x8297D348ACCAD23A,
    0xB6CF4511EF22F42D,
    0x77781C3F72A16C48,
    0x0000001B83B4F3FC,
    0x41ECE66B35DC377B,
    0xCD891EEA95D9B170,
    0xD8E232CA6E18FB66,
    0xB477E51CE77DBB6D,
    0xC2AC152882E5FE0C,
    0x0BF841E520AAAB2E,
    0x85664F67D1F77E4D,
    0xF16B03797F724C9D,
    0x00000164A6535B7F,
    0xEB97106B875A7881,
    0x62AEB3A186D96739,
    0x418981AB8729E006,
    0x7B160262DEF8AF19,
    0x2F8AE98690437D75,
    0xAC749CE2C3B04E07,
    0x4BFACF08C406FE1E,
    0x986108099440A43F,
    0x000184F537E7AEBB,
    0x39AB7AA0F04EC905,
    0xEC59208E7DB908F8,
    0x135EEC11925D16C6,
    0xD48CB7FAE63EA812,
    0x3C5EBDE5B1698940,
    0x175F4658F1BAE9B1,
    0x261765F22251A78F,
    0x574D66DE912BDBB8,
    0xFA208C8FD9A80F6D,
    0x0000000000000037,
    0xBDB61B56F4BA0B45,
    0x505D53695B020710,
    0x754563E9B2547DD5,
    0x0A1411A2AFA2D5DD,
    0x3E397AD8FD028D08,
    0x4892132690B33971,
    0xB18B8894D921E971,
    0x4B942EDB28568CCD,
    0xCA2E631107AFC7E8,
    0x28F12288B02990D5,
    0x0431985AD09F54E6,
    0xE1E36C6DE442C511,
    0xB80103285F85D56E,
    0x83DC3BA219D442BA,
    0xDBD88112373971C6,
    0x328305A2A6E0D1EC,
    0xE74B816F950CEFB3,
    0xE2E11F7BCA0FFBB5,
    0x000000000000B4BC,
    0xBCB022FD2336600B,
    0x1C7B52F195E65F90,
    0xB9E15922C7E230BC,
    0x2F59A9817DD41133,
    0x7D107A383AB6B5D6,
    0x94F6F3DADED91BEF,
    0x6A0BB1F465A92D18,
    0x20A47D220FB190AA,
    0xA089AE8CBAAC2C6A,
    0x00000000010106E6,
    0x0969B17FBC6A1A9F,
    0x33F487F8D72F5693,
    0xD4F5AD5B4B1E47D1,
    0xD52985FAB0CC5457,
    0xCE67B56CD8A88625,
    0x2742A8E80F9ED527,
    0xB5842236221F6F60,
    0x47664FA5713B3FEC,
    0xA2A171DEE48C0420,
    0x0000050911DF8831,
    0x67CFFB9A3910FBAD,
    0x1BC3808BA0685677,
    0x8AC634279D37EC22,
    0x11E009C7BB32428E,
    0x3081ABCDE519922B,
    0x2ACED3E296483266,
    0x8F72727322A27FD4,
    0x953982F35CB9B688,
    0xCB82CAC03B13C06A,
    0x0000042257ADBA56,
    0x2558D03DCD08A241,
    0xC5FF8DB676901647,
    0xB70EC13A7298B59C,
    0x3720BCDF8919C115,
    0xB89F0AB1FFA0D860,
    0x19EF24140DCE37E8,
    0x4A770587EEDB6FEF,
    0x643530792D5C888C,
    0x635520660F5C524B,
    0xD1BB232A7D6771D4,
    0x000000000014A549,
    0x8AFFDF05439A96F7,
    0x22066CA1ACD1EA3E,
    0xFF4A0F80470DA1B7,
    0x09E00EC582D9F671,
    0x79C3558D339D644F,
    0xE31A822338AA7E02,
    0xC45885C990C69413,
    0xDE712E4CD3A09971,
    0xE8630251C240838D,
    0x003D709224784443,
    0xD46638028EBEFE4F,
    0xFD39B1DE1583FF01,
    0xF706CC3E01AD06BF,
    0x018200EEDDDCF204,
    0xD02C90CC5F35974F,
    0x0A4D510AF2F7400F,
    0x01E72BA3292622CB,
    0x545627EE29B953C7,
    0x7EDCFAE6363A2FC8,
    0xE892E4516AD3CA1F,
    0x00000000000000BB,
    0x16A2278EBE4BF829,
    0x13F71FD2BCA1D7CE,
    0x13B13EC424981522,
    0xFC90EE82E49171A8,
    0x0AB83057F38CA1CF,
    0xC3AB52BD6DF95DE3,
    0x56CBB597E7A825BB,
    0xC25F61ECFB567CB3,
    0xF30BCED6C6827E4E,
    0xABBD93FEDD979774,
    0x000000000000130E,
    0x8CBE87BCA10C14E9,
    0x3278301BE173D18D,
    0x7509B5965C5E5D23,
    0x01725CE6A87E389C,
    0xFCDC281B3C43B235,
    0x63F803B5B92CD801,
    0x57A311B688269122,
    0xC74FC45E6F177C62,
    0x73A76D7AE92EA197,
    0x8DD0D0193D34E4A2,
    0x00000000002F59F8,
    0xD4578F8B1ADAEC13,
    0x52C4FCCD96F870A9,
    0xA9CE8F994E7739D3,
    0xB5B1903B254AF9C6,
    0x53B9AB5C27CEAFD6,
    0xD07DBF8A1BE32413,
    0x82DEEC929D53A7A3,
    0xDCCC701DB788C03D,
    0xCA717EC32E54D33F,
    0xF495A8FAD4DA1277,
    0x0000011429A84B24,
    0x31556B4BF5EC283B,
    0x49C734DE4115002D,
    0xB259CE4FF95AECC8,
    0x63E7153D64DEB815,
    0xDAB88DCE4161BD53,
    0x8484201669B8D719,
    0x4438DD9DE0706C3B,
    0x43203CA26C066FB3,
    0xC609E4352DB3A68A,
    0xB35CEA3AA52E1357,
    0x1DDC6A148261AB9B,
    0x0000000000000004,
    0xC8F6726DEB962C7F,
    0x90F7C1D6E50BF9FC,
    0x911276B8688D3044,
    0xDF72CB9398C7A082,
    0x7192F9DB2CF0A6CB,
    0x93450018EFA5557D,
    0x71F4F05B4BB291D8,
    0x3AC05F4F9216F585,
    0x858979A501FBD4D6,
    0x71ECE31ED275AFA2,
    0x0001C4593A878336,
    0xE79650C8587A5149,
    0xB18E44DE74979FDA,
    0x632E8C3372D034FB,
    0x1DBCFB18CD54F832,
    0xCFDF90BC92D49F6D,
    0xA4CE9EB6C21DFA1A,
    0x90FDE09E4C993EC1,
    0x1CC426F7B83F4CCF,
    0xF06E5B7BE8A96BDF,
    0x164BBC62A3FBFC3A,
    0x9F2124C549104736,
    0x00000000000002E9,
    0x2FA931941A16C039,
    0x6AEDEF17A1B86686,
    0xE0CEEDDE8FA4C8A8,
    0x23CC4EE1E9F44A7E,
    0x4BEEBFF978709303,
    0x13BDEFA1ADE4B3C6,
    0x74B621026F66010D,
    0x86EE8EBEE828E3FC,
    0xBC564A1AA88784A6,
    0xFEA50F8E9F3A98F7,
    0x075E0F03567164C9,
    0x000000003EB24EB8,
    0x70B5AB76220044EF,
    0xBB00314100411E63,
    0x827A013DF1803754,
    0xB65FFA77992C9800,
    0xF1D568EDC6802120,
    0x09287CAE2BBBE5F0,
    0x960333E6C6106646,
    0x9EB1A318D308592C,
    0x43861FACB9F63B00,
    0x9DECDCF84B1C0292,
    0x26C68B0752877B46,
    0x0000000160F62273,
];

/// Where each numerator of [`LONG`] ends.
const LONG_END: [u16; 71] = [
    3, 5, 8, 11, 14, 17, 20, 23, 26, 29, 33, 37, 41, 45, 49, 53, 57, 61, 66, 70, 75, 80, 85, 90,
    96, 101, 107, 113, 119, 125, 131, 137, 143, 150, 157, 164, 171, 178, 185, 192, 200, 207, 215,
    223, 231, 239, 247, 255, 264, 273, 282, 291, 300, 309, 319, 328, 338, 348, 358, 368, 379, 389,
    400, 411, 422, 433, 445, 456, 468, 480, 492,
];

/// The denominators of `B_60, B_62, ..., B_200`.
const LONG_DEN: [u64; 71] = [
    56786730,
    6,
    510,
    64722,
    30,
    4686,
    140100870,
    6,
    30,
    3318,
    230010,
    498,
    3404310,
    6,
    61410,
    272118,
    1410,
    6,
    4501770,
    6,
    33330,
    4326,
    1590,
    642,
    209191710,
    1518,
    1671270,
    42,
    1770,
    6,
    2328255930,
    6,
    30,
    4357878,
    510,
    8646,
    4206930,
    6,
    4110,
    274386,
    679470,
    6,
    2381714790,
    6,
    4470,
    2162622,
    30,
    138,
    1794590070,
    6,
    230010,
    130074,
    2490,
    1002,
    3404310,
    66,
    5190,
    2478,
    1043970,
    1074,
    7225713885390,
    6,
    1410,
    42,
    30,
    12606,
    868841610,
    6,
    171390,
    244713882,
    1366530,
];

/// The last index of the tables.
const TABLED: u64 = 200;

/// `B_n` for an even `n` up to [`TABLED`], from the tables, as numbers (in words they are taken
/// inline by [`bernoulli`]).
fn tabled(n: u64, order: ByteOrder) -> Rational {
    if n / 2 < TABLE.len() as u64 {
        let (a, b) = TABLE[(n / 2) as usize];
        return Rational::from_reduced(
            number_u128(a.unsigned_abs(), a < 0, order),
            number_u128(u128::from(b), false, order),
        );
    }
    let i = ((n - 2 * TABLE.len() as u64) / 2) as usize;
    let start = if i == 0 {
        0
    } else {
        usize::from(LONG_END[i - 1])
    };
    Rational::from_reduced(
        to_number(
            &LONG[start..usize::from(LONG_END[i])],
            n.is_multiple_of(4),
            order,
        ),
        number_u128(u128::from(LONG_DEN[i]), false, order),
    )
}

/// `B_n`, in the byte order `order`: those in words from the tables inline, in the caller, with
/// nothing to save or allocate, and the rest by [`beyond_words`].
#[inline]
pub(super) fn bernoulli(n: u64, order: ByteOrder) -> Rational {
    if n == 1 {
        return Rational::from_word(true, 1, 2, order);
    }
    if n & 1 == 1 {
        return Rational::from_word(false, 0, 1, order);
    }
    match word_of(n) {
        Some((negative, m, d)) => Rational::from_word(negative, m, d, order),
        None => beyond_words(n, order),
    }
}

/// The sign and the parts of an even `B_n` when it is in [`TABLE`] and they fit in 64 bits (all to
/// `B_34`, and `B_38`).
#[inline(always)]
fn word_of(n: u64) -> Option<(bool, u64, u64)> {
    let (a, b) = *TABLE.get(usize::try_from(n / 2).ok()?)?;
    let m = a.unsigned_abs();
    (m >> 64 == 0).then_some((a < 0, m as u64, b))
}

/// An even `B_n` that is not in words: from the tables up to [`TABLED`], and beyond by the zeta
/// function, out of line so that the lookups keep no registers for it.
#[inline(never)]
fn beyond_words(n: u64, order: ByteOrder) -> Rational {
    if n <= TABLED {
        return tabled(n, order);
    }
    let (num, den) = by_zeta(n);
    Rational::from_reduced(
        to_number(&num, n.is_multiple_of(4), order),
        to_number(&den, false, order),
    )
}

/// The denominator of `B_n` for an even `n >= 2`, by von Staudt and Clausen's theorem: the product
/// of the primes `p` with `p - 1` dividing `n`.
pub(super) fn denominator(n: u64) -> Limbs {
    let mut f: Vec<u64> = Vec::new();
    let mut packed = 1u64;
    each_divisor(n, |d| {
        if let Some(p) = d.checked_add(1)
            && is_prime_u64(p)
        {
            match packed.checked_mul(p) {
                Some(x) => packed = x,
                None => {
                    f.push(packed);
                    packed = p;
                }
            }
        }
    });
    f.push(packed);
    product(&f)
}

/// Calls `f` with each divisor of `n > 0`, in no particular order.
fn each_divisor(n: u64, mut f: impl FnMut(u64)) {
    // The prime factors by trial division; `n` stays small for any Bernoulli number that fits in
    // memory.
    let mut factors: Vec<(u64, u32)> = Vec::new();
    let mut m = n;
    let mut d = 2u64;
    while d * d <= m {
        if m.is_multiple_of(d) {
            let mut e = 0;
            while m.is_multiple_of(d) {
                m /= d;
                e += 1;
            }
            factors.push((d, e));
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if m > 1 {
        factors.push((m, 1));
    }
    let mut divisors = vec![1u64];
    for &(p, e) in &factors {
        let len = divisors.len();
        let mut q = 1;
        for _ in 0..e {
            q *= p;
            for i in 0..len {
                divisors.push(divisors[i] * q);
            }
        }
    }
    for d in divisors {
        f(d);
    }
}

/// A positive floating number `m 2^e`.
#[derive(Clone)]
struct Float {
    m: Limbs,
    e: i64,
}

impl Float {
    /// The number cut to its `w` leading bits, rounded down: within `2^(1 - w)` of it relatively.
    fn cut(mut self, w: u64) -> Float {
        self.cut_in(w);
        self
    }

    /// [`cut`](Self::cut) in place.
    fn cut_in(&mut self, w: u64) {
        let b = bits(&self.m);
        if b > w {
            shr_assign(&mut self.m, b - w);
            self.e += (b - w) as i64;
        }
    }

    /// `x <- x y`, cut to `w` bits, the product in the working memory.
    fn mul_in(&mut self, y: &Float, w: u64, work: &mut Work) {
        mul_into(&mut work.t, &self.m, &y.m, &mut work.s);
        normalize(&mut work.t);
        core::mem::swap(&mut self.m, &mut work.t);
        self.e += y.e;
        self.cut_in(w);
    }

    /// `x <- x^2`, cut to `w` bits, the square in the working memory.
    fn sqr_in(&mut self, w: u64, work: &mut Work) {
        sqr_into(&mut work.t, &self.m, &mut work.s);
        normalize(&mut work.t);
        core::mem::swap(&mut self.m, &mut work.t);
        self.e *= 2;
        self.cut_in(w);
    }
}

/// The vectors that the products and quotients of one computation reuse, so that a short one
/// allocates a few times rather than at every step.
#[derive(Default)]
struct Work {
    /// A product, swapped with the operand it replaces.
    t: Limbs,
    /// The scratch of the longer products.
    s: Limbs,
    /// A shifted operand of a quotient.
    a: Limbs,
    /// The normalised dividend and divisor of a quotient.
    n: Limbs,
    d: Limbs,
}

/// `x^n` from the top bit of `n`, each square and product cut to `w` bits: within
/// `4 n 2^(1 - w)` relatively, to the first order, of `x'^n` for the `x'` given.
fn power(x: &Float, n: u64, w: u64, work: &mut Work) -> Float {
    let mut acc = Float {
        m: Vec::with_capacity(2 * w.div_ceil(64) as usize + 2),
        e: 0,
    };
    power_into(&mut acc, x, n, w, work);
    acc
}

/// [`power`] into `acc`, whose vector is reused.
fn power_into(acc: &mut Float, x: &Float, n: u64, w: u64, work: &mut Work) {
    let small = x.e == 0 && x.m.len() == 1 && bits(&x.m) <= w;
    acc.m.clear();
    acc.m.extend_from_slice(&x.m);
    acc.e = x.e;
    acc.cut_in(w);
    for i in (0..63 - n.leading_zeros()).rev() {
        acc.sqr_in(w, work);
        if n >> i & 1 == 1 {
            if small {
                mul_1(&mut acc.m, x.m[0]);
                acc.cut_in(w);
            } else {
                acc.mul_in(x, w, work);
            }
        }
    }
}

/// Up to this many limbs in the divisor, a quotient is taken by the schoolbook method in the
/// working memory; beyond, on numbers, whose division is faster from there. Measured on an Apple
/// M4 Max, `B_300` took 18,400 cycles with 8, 19,400 with 16 and 20,800 with 32 (4 as 8), `B_600`
/// 56,600, 62,700 and 65,800, and `B_1000` 157,000, 156,000 and 167,000.
const SCHOOL_DIVISION: usize = 8;

/// `q = floor(a 2^s / b)` for magnitudes, `b` not zero and `s` of either sign (`b 2^-s` when
/// negative).
fn quotient_into(q: &mut Limbs, a: &[u64], s: i64, b: &[u64], work: &mut Work) {
    let (num, den): (&[u64], &[u64]) = if s >= 0 {
        shl_into(&mut work.a, a, s as u64);
        (&work.a, b)
    } else {
        shl_into(&mut work.a, b, s.unsigned_abs());
        (a, &work.a)
    };
    if den.len() > SCHOOL_DIVISION {
        let o = ByteOrder::LittleEndian;
        *q = of_number(&(&to_number(num, false, o) / &to_number(den, false, o)));
        return;
    }
    divide_into(q, num, den, &mut work.n, &mut work.d);
}

/// `q = floor(a / b)` by Knuth's algorithm D (volume 2, section 4.3.1), `b` normalised in `nd` and
/// `a` shifted as much in `nn`.
fn divide_into(q: &mut Limbs, a: &[u64], b: &[u64], nn: &mut Limbs, nd: &mut Limbs) {
    q.clear();
    let m = b.len();
    if a.len() < m {
        return;
    }
    if m == 1 {
        q.extend_from_slice(a);
        super::limbs::div_1(q, b[0]);
        return;
    }
    let sh = b[m - 1].leading_zeros();
    shl_into(nd, b, u64::from(sh));
    nd.truncate(m);
    shl_into(nn, a, u64::from(sh));
    nn.resize(a.len() + 1, 0);
    let len = a.len() - m + 1;
    q.resize(len, 0);
    let (top, next) = (u128::from(nd[m - 1]), u128::from(nd[m - 2]));
    let v = invert_limb(nd[m - 1]);
    for j in (0..len).rev() {
        // The estimate from the top two limbs, by the reciprocal of the divisor's top limb.
        let (u1, u0) = (nn[j + m], nn[j + m - 1]);
        let (mut qhat, mut rhat) = if u1 >= nd[m - 1] {
            (u128::from(u64::MAX), u128::from(u0) + top)
        } else {
            let (q, r) = div_2by1(u1, u0, nd[m - 1], v);
            (u128::from(q), u128::from(r))
        };
        while rhat >> 64 == 0 && qhat * next > ((rhat << 64) | u128::from(nn[j + m - 2])) {
            qhat -= 1;
            rhat += top;
        }
        // nn[j..=j + m] -= qhat nd.
        let (mut borrow, mut carry) = (0u64, 0u64);
        for i in 0..m {
            let p = qhat * u128::from(nd[i]) + u128::from(carry);
            carry = (p >> 64) as u64;
            let (d1, b1) = nn[j + i].overflowing_sub(p as u64);
            let (d2, b2) = d1.overflowing_sub(borrow);
            nn[j + i] = d2;
            borrow = u64::from(b1 | b2);
        }
        let (d1, b1) = nn[j + m].overflowing_sub(carry);
        let (d2, b2) = d1.overflowing_sub(borrow);
        nn[j + m] = d2;
        if b1 | b2 {
            // One too many: add the divisor back.
            qhat -= 1;
            let mut c = false;
            for i in 0..m {
                let (s1, c1) = nn[j + i].overflowing_add(nd[i]);
                let (s2, c2) = s1.overflowing_add(u64::from(c));
                nn[j + i] = s2;
                c = c1 | c2;
            }
            nn[j + m] = nn[j + m].wrapping_add(u64::from(c));
        }
        q[j] = qhat as u64;
    }
    normalize(q);
}

/// `log2(2π)`, rounded up.
const LOG2_TWO_PI: f64 = 2.651_496_129_472_319;

/// The bits of `n`.
fn bit_length(n: u64) -> u64 {
    u64::from(64 - n.leading_zeros())
}

/// `floor(π 2^f)` less at most 4: up to 16,384 bits cut from the table of the partition numbers,
/// beyond from their Chudnovsky series.
fn pi_fixed(f: u64) -> Limbs {
    let table = 64 * PI_TABLE.len() as u64;
    if f <= table {
        let mut x = shr(PI_TABLE, table - f);
        super::limbs::add(&mut x, &shl(&[3], f));
        x
    } else {
        of_number(&pi(f))
    }
}

/// `(|B_n| d, d)` for an even `n` above the table, `d` the denominator, from
/// `|B_n| = 2 n! ζ(n) / (2π)^n`: `|B_n| d` computed within less than 1/2 and rounded, or, for long
/// numerators, its low bits modulo primes by Harvey's method ([`modp`](super::modp)) and the rest
/// from the formula, to as many bits fewer.
fn by_zeta(n: u64) -> (Limbs, Limbs) {
    let o = ByteOrder::LittleEndian;
    let f = of_number(&Number::factorial(n, o));
    let den = denominator(n);
    let k = shl(&mul(&f, &den), 1);
    drop(f);
    // log2(|B_n| d) < bits(K) + log2 ζ(2) - n log2(2π), log2 ζ(2) < 0.73.
    let u = (bits(&k) as f64 + 1.75 - n as f64 * LOG2_TWO_PI)
        .ceil()
        .max(1.0) as u64
        + 1;
    let share = modular_bits(n, u);
    if share == 0 {
        let mut q = approximation(n, k, u + 8);
        super::limbs::add(&mut q, &[8]);
        return (shr(&q, 4), den);
    }
    // |B_n| d modulo the primes, joined: `R = |B_n| d mod M`.
    let (r, m) = modular(n, share, &den);
    // |B_n| d within M / 2^11, and the multiple of M nearest to its difference from R.
    let w = u.saturating_sub(bits(&m)) + 12;
    let q = approximation(n, k, w);
    let a = to_number(&shr(&q, 4), false, o);
    let (rn, mn) = (to_number(&r, false, o), to_number(&m, false, o));
    let t = (&(&a - &rn) + &(&mn >> 1usize)).div_floor(&mn);
    let x = &rn + &(&t * &mn);
    (of_number(&x), den)
}

/// The bits of `|B_n| d`, of `u` at most, taken modulo primes: none below [`MODULAR`], then a share
/// `α = 0.1 + 0.05 log2(n / 10^4)` that grows with `n` up to 0.22. Measured on an Apple M4 Max:
/// `B_5000` took 0.98 ms with none and 1.03 with 5 %; `B_10^4` 3.86 ms with none, 3.55 with 10 % and
/// 3.59 with 20 %; `B_(2 10^4)` 16.0 ms, 11.3, 10.4 with 15 % and 10.4; `B_(4 10^4)` 56.6, 36.0, 32.0
/// and 31.0 with 20 %; `B_10^5` 362 ms with none and 172, 157, 160 and 172 with 15, 20, 25 and 30 %;
/// `B_10^6` 9.24 s, 8.02, 8.09 and 9.13 with the same.
fn modular_bits(n: u64, u: u64) -> u64 {
    if n < MODULAR {
        return 0;
    }
    let alpha = (0.1 + 0.05 * (n as f64 / 1e4).log2()).clamp(0.0, 0.22);
    (u as f64 * alpha) as u64
}

/// The index from which the low bits of a Bernoulli number's numerator are taken modulo primes.
const MODULAR: u64 = 8000;

/// `(|B_n| d mod M, M)` for `M` the product of primes from 5 up, of at least `want` bits, those that
/// divide `d` or for which Harvey's method does not apply left out.
fn modular(n: u64, want: u64, den: &[u64]) -> (Limbs, Limbs) {
    let o = ByteOrder::LittleEndian;
    // The primes up to a bound that their bits reach (the product of the primes up to x has about
    // x / ln 2 bits), and more if needed.
    let mut limit = ((want as f64) * core::f64::consts::LN_2 * 1.1) as u64 + 1000;
    loop {
        let primes = odd_primes(limit);
        let root = (limit as f64).sqrt() as u64 + 2;
        let small: Vec<u64> = primes.iter().copied().take_while(|&q| q <= root).collect();
        let mut congruences: Vec<(Number, Number)> = Vec::new();
        let mut got = 0.0f64;
        let negative = n.is_multiple_of(4);
        for &p in primes.iter().filter(|&&p| p >= 5) {
            if let Some(b) = super::modp::bernoulli_mod_p(n, p, &small) {
                // |B_n| d mod p.
                let b = if negative { (p - b) % p } else { b };
                let r = (u128::from(b) * u128::from(super::limbs::rem_1(den, p)) % u128::from(p))
                    as u64;
                congruences.push((Number::new_from_u64(r, o), Number::new_from_u64(p, o)));
                got += (p as f64).log2();
                if got >= want as f64 {
                    let (x, m) = Number::crt(&congruences).expect("the primes are coprime");
                    return (of_number(&x), of_number(&m));
                }
            }
        }
        limit *= 2;
    }
}

/// `|B_n| d 2^4`, for `K = 2 d n!`, within `2^(u - w + 5)` units for `u` a bound of the bits of
/// `|B_n| d`, from `|B_n| d = K ζ(n) / (2π)^n` computed to `w` bits relatively: `(2π)^n` by powers
/// cut to `w + log2 n + 6` bits from `π` to as many, and `1/ζ(n) = Π_p (1 - p^-n)` in fixed point
/// with `w + log2 n + 4` bits, each `p^-n` to the bits it moves the product by. The errors add up
/// to less than `2^-w` relatively; the quotient is taken with 4 bits below the point.
fn approximation(n: u64, k: Limbs, w: u64) -> Limbs {
    let nl = bit_length(n);
    let mut work = Work::default();
    // (2π)^n.
    let wr = w + nl + 6;
    let two_pi = Float {
        m: pi_fixed(wr),
        e: 1 - wr as i64,
    };
    let r = power(&two_pi, n, wr, &mut work);
    // 1/ζ(n) 2^wz.
    let wz = w + nl + 4;
    let z = inverse_zeta(n, wz, &mut work);
    // (2π)^n / ζ(n), as `dm 2^de`.
    let mut dm = Float {
        m: z,
        e: -(wz as i64),
    };
    dm.mul_in(&r, w + 8, &mut work);
    let km = Float { m: k, e: 0 }.cut(w + 8);
    // |B_n| d 2^4 = (km / dm) 2^(ke - de + 4).
    let mut q = Vec::new();
    quotient_into(&mut q, &km.m, km.e - dm.e + 4, &dm.m, &mut work);
    q
}

/// `floor(2^w / ζ(n))` within a few units for every prime it takes, from the Euler product
/// `1/ζ(n) = Π_p (1 - p^-n)` over the primes whose `2^w p^-n` is at least 1: `z <- z - z / p^n`,
/// `z` and `p^n` cut to the bits of `2^w p^-n` and `log2 n + 8` more, one division.
fn inverse_zeta(n: u64, w: u64, work: &mut Work) -> Limbs {
    let nl = bit_length(n);
    // 2^w (1 - 2^-n).
    let mut z = shl(&[1], w);
    if n <= w {
        sub(&mut z, &shl(&[1], w - n));
    }
    let mut pn = Float {
        m: Vec::with_capacity(z.len() + 2),
        e: 0,
    };
    let (mut zt, mut delta) = (Vec::with_capacity(z.len()), Vec::with_capacity(z.len()));
    // The primes with `p^n <= 2^(w + 1)`.
    let limit = ((w + 1) as f64 / n as f64).exp2() as u64 + 2;
    for p in primes(3..=limit) {
        let lg = n as f64 * (p as f64).log2();
        if lg > w as f64 + 1.0 {
            break;
        }
        let t = (w as f64 - lg).floor().max(0.0) as u64 + 2;
        let prec = t + nl + 8;
        // z / p^n = (zt 2^(w - prec)) / (m 2^e).
        let cut = w.saturating_sub(prec);
        shr_into(&mut zt, &z, cut);
        power_into(&mut pn, &Float { m: vec![p], e: 0 }, n, prec, work);
        quotient_into(&mut delta, &zt, cut as i64 - pn.e, &pn.m, work);
        if delta.is_empty() {
            break;
        }
        sub(&mut z, &delta);
    }
    z
}

/// `B_0, ..., B_n`, in the byte order `order`.
pub(super) fn bernoulli_numbers(n: u64, order: ByteOrder) -> Vec<Rational> {
    let len = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_add(1))
        .expect("capacity overflow");
    let mut v: Vec<Rational> = Vec::with_capacity(len);
    // The tables', words written in place, and zeros where the others go.
    for k in 0..=n.min(TABLED) {
        v.push(bernoulli(k, order));
    }
    for _ in TABLED.min(n)..n {
        v.push(Rational::from_word(false, 0, 1, order));
    }
    let top = n & !1;
    let low = TABLED + 2;
    if top >= low {
        table_by_zeta(top, low, |k, num, den| {
            v[k as usize] = Rational::from_reduced(
                to_number(&num, k.is_multiple_of(4), order),
                to_number(&den, false, order),
            );
        });
    }
    v
}

/// `log2 x` for a magnitude `x` of at least one limb, from its top 64 bits.
fn log2_of(x: &[u64]) -> f64 {
    let b = bits(x);
    let top = shr(x, b.saturating_sub(64));
    let t = top.first().copied().unwrap_or(0) as f64;
    t.log2() + b.saturating_sub(64) as f64
}

/// `(|B_k| d_k, d_k)` for every even `k` from `top` down to `low`, handed to `put`, by the formula of
/// [`by_zeta`] run from the top index down, as FLINT's `bernoulli_rev` does: `G_k = 2 k! / (2π)^k`
/// from the last by a product with `(2π)^2` and a division by `k (k - 1)`, and
/// `ζ(k) = (1 - 2^-k)^-1 Σ_(j odd) j^-k` from the powers `j^-k` in fixed point, each from the last by
/// a product with the limb `j^2`. Both are kept to the bits of `|B_k| d_k` and `log2 top + 8` more,
/// which fall as `k` falls, the powers to whole limbs and a limb more, and the bounds of their errors
/// in units of their last bits; a number whose bound would not leave less than 1/4 is computed
/// alone, which the bounds do not let happen.
fn table_by_zeta(top: u64, low: u64, mut put: impl FnMut(u64, Limbs, Limbs)) {
    let nl = bit_length(top);
    let g = nl + 8;
    let fact = of_number(&Number::factorial(top, ByteOrder::LittleEndian));
    let mut lf = log2_of(&fact);
    // The bits of `|B_k| d_k`, bounded from `log2(2 d_k k!) + log2 ζ(2) - k log2(2π)` with 3 bits
    // for the rounding of the doubles.
    let bound = |k: u64, lf: f64, ld: f64| -> u64 {
        (1.0 + ld + lf + 0.75 - k as f64 * LOG2_TWO_PI)
            .ceil()
            .max(1.0) as u64
            + 3
    };
    // The denominators, and the largest of their bits up to each index, which keeps the precisions
    // falling with `k`.
    let mut dens: Vec<Limbs> = (low / 2..=top / 2).map(|i| denominator(2 * i)).collect();
    let mut most: Vec<f64> = Vec::with_capacity(dens.len());
    let mut m = 0.0f64;
    for d in &dens {
        m = m.max(bits(d) as f64);
        most.push(m);
    }
    let at = |k: u64| ((k - low) / 2) as usize;
    let mut u = bound(top, lf, most[at(top)]);
    // G to `p` bits; the powers in fixed point with `64 l` bits, `l` limbs.
    let mut p = u + 8 + g;
    let mut l = (p + 64).div_ceil(64) as usize;
    let wr = p + nl + 6;
    let two_pi = Float {
        m: pi_fixed(wr),
        e: 1 - wr as i64,
    };
    let mut work = Work::default();
    let mut four_pi2 = two_pi.clone();
    four_pi2.sqr_in(wr, &mut work);
    let r = power(&two_pi, top, wr, &mut work);
    let km = Float {
        m: shl(&fact, 1),
        e: 0,
    }
    .cut(p + 8);
    drop(fact);
    let s = km.e - r.e + p as i64 + 8 - bits(&km.m) as i64 + bits(&r.m) as i64;
    let mut gm = Float {
        m: {
            let mut q = Vec::new();
            quotient_into(&mut q, &km.m, s, &r.m, &mut work);
            q
        },
        e: km.e - r.e - s,
    }
    .cut(p);
    // The bound of G's error, in units of `2^-p` relatively.
    let mut err_g = 8.0f64;
    // The odd powers: `ys[i] = floor(2^(64 l) (2i + 3)^-k)`, the bounds of their errors in those
    // units, and their sum with 1.
    let mut ys: Vec<Limbs> = Vec::new();
    let mut err_y: Vec<f64> = Vec::new();
    let mut sum: Limbs = vec![0; l + 2];
    sum[l] = 1;
    let mut k = top;
    loop {
        let units = 64 * l as u64;
        // More powers while the tail beyond them could reach half a unit.
        let tail_of = |j0: u64, k: u64| {
            units as f64 - k as f64 * (j0 as f64).log2()
                + (1.0 + j0 as f64 / (2.0 * (k - 1) as f64)).log2()
        };
        while tail_of(2 * ys.len() as u64 + 3, k) >= -1.0 {
            let y = power_reciprocal(2 * ys.len() as u64 + 3, k, units, nl, &mut work);
            add_into(&mut sum, &y);
            ys.push(y);
            err_y.push(2.0);
        }
        let tail = tail_of(2 * ys.len() as u64 + 3, k).exp2();
        // ζ(k) 2^units = sum (1 + 2^-k + 2^-2k + ...).
        let mut zeta = sum.clone();
        normalize(&mut zeta);
        let mut i = k;
        while i <= units {
            add_into(&mut zeta, &shr(&sum, i));
            i += k;
        }
        normalize(&mut zeta);
        let err_z = err_y.iter().sum::<f64>() + tail + (units / k + 1) as f64;
        // |B_k| d_k 2^4 = d_k G ζ 2^(e + 4 - units).
        let d = core::mem::take(&mut dens[at(k)]);
        let x = mul(&mul(&gm.m, &zeta), &d);
        let sh = gm.e - units as i64 + 4;
        let mut q = if sh >= 0 {
            shl(&x, sh as u64)
        } else {
            shr(&x, sh.unsigned_abs())
        };
        // The error bound, in units of the result: `2^u (err_g 2^-p + err_z 2^-units)`, and the
        // floors.
        let err = (err_g * (u as f64 - p as f64).exp2()) + err_z * (u as f64 - units as f64).exp2();
        if err + 1.0 / 8.0 < 0.25 {
            super::limbs::add(&mut q, &[8]);
            put(k, shr(&q, 4), d);
        } else {
            let (num, den) = by_zeta(k);
            put(k, num, den);
        }
        if k <= low {
            break;
        }
        // Down to k - 2.
        lf -= ((k * (k - 1)) as f64).log2();
        let k2 = k - 2;
        u = bound(k2, lf, most[at(k2)]);
        let p2 = u + 8 + g;
        let l2 = (p2 + 64).div_ceil(64) as usize;
        let dl = l - l2;
        // G_(k-2) = G_k (2π)^2 / (k (k - 1)), `(2π)^2` cut to the limbs that count.
        let fp = four_pi2
            .m
            .len()
            .saturating_sub(p2.div_ceil(64) as usize + 2);
        let mut m = mul(&gm.m, &four_pi2.m[fp..]);
        let e = gm.e + four_pi2.e + 64 * fp as i64;
        let b = bits(&m);
        let c = b.saturating_sub(p2 + 64);
        m = shr(&m, c);
        super::limbs::div_1(&mut m, k * (k - 1));
        gm = Float { m, e: e + c as i64 }.cut(p2);
        // Errors in units of 2^-p2: the old ones shrink by 2^-(p - p2), `(2π)^2` and its cut add
        // theirs (within `6 2^-wr` and `2^(1 - 64 (len - fp) + 64)`), the cut and the division a
        // unit each.
        let kept = 64.0 * (four_pi2.m.len() - fp) as f64 - 64.0;
        err_g = err_g * (p2 as f64 - p as f64).exp2()
            + 6.0 * (p2 as f64 - wr as f64).exp2()
            + 2.0 * (p2 as f64 - kept).exp2()
            + 3.0;
        // y_j (k - 2) = y_j(k) j^2 B^-dl, summed as they are made.
        sum.clear();
        sum.resize(l2 + 2, 0);
        sum[l2] = 1;
        for (i, y) in ys.iter_mut().enumerate() {
            let j = 2 * i as u64 + 3;
            scale_into(y, j * j, dl, &mut sum);
            err_y[i] = err_y[i] * (j * j) as f64 * (-64.0 * dl as f64).exp2() + 1.0;
        }
        while ys.last().is_some_and(|y| y.is_empty()) {
            ys.pop();
            err_y.pop();
        }
        p = p2;
        l = l2;
        k = k2;
    }
}

/// `s + y` in place, `s` long enough for the sum.
#[inline]
fn add_into(s: &mut Limbs, y: &[u64]) {
    if s.len() <= y.len() {
        s.resize(y.len() + 1, 0);
    }
    let mut carry = 0u64;
    let (low, high) = s.split_at_mut(y.len());
    for (a, &b) in low.iter_mut().zip(y) {
        let (t, c1) = a.overflowing_add(b);
        let (t, c2) = t.overflowing_add(carry);
        *a = t;
        carry = u64::from(c1 | c2);
    }
    for a in high.iter_mut() {
        if carry == 0 {
            break;
        }
        let (t, c) = a.overflowing_add(carry);
        *a = t;
        carry = u64::from(c);
    }
}

/// `y <- floor(y c / B^drop)` and `s += y`, in one pass over the limbs of `y`; `s` long enough.
#[inline]
fn scale_into(y: &mut Limbs, c: u64, drop: usize, s: &mut Limbs) {
    let len = y.len();
    if drop > len {
        y.clear();
        return;
    }
    let mut carry = 0u64;
    // The product's limbs below `drop` only carry.
    for &x in &y[..drop.min(len)] {
        let t = u128::from(x) * u128::from(c) + u128::from(carry);
        carry = (t >> 64) as u64;
    }
    let mut sc = false;
    let mut out = 0;
    for i in drop..len {
        let t = u128::from(y[i]) * u128::from(c) + u128::from(carry);
        let v = t as u64;
        carry = (t >> 64) as u64;
        y[out] = v;
        let (a, c1) = s[out].overflowing_add(v);
        let (a, c2) = a.overflowing_add(u64::from(sc));
        s[out] = a;
        sc = c1 | c2;
        out += 1;
    }
    if carry != 0 {
        if out < len {
            y[out] = carry;
        } else {
            y.push(carry);
        }
        let (a, c1) = s[out].overflowing_add(carry);
        let (a, c2) = a.overflowing_add(u64::from(sc));
        s[out] = a;
        sc = c1 | c2;
        out += 1;
    }
    y.truncate(out);
    let mut i = out;
    while sc {
        let (a, c) = s[i].overflowing_add(1);
        s[i] = a;
        sc = c;
        i += 1;
    }
    normalize(y);
}

/// `floor(2^p j^-k)` within 2, by `j^k` cut to the bits of the result and `log2 k + 8` more, and a
/// division.
fn power_reciprocal(j: u64, k: u64, p: u64, nl: u64, work: &mut Work) -> Limbs {
    let lg = k as f64 * (j as f64).log2();
    let t = (p as f64 - lg).floor().max(0.0) as u64 + 2;
    let prec = t + nl + 8;
    let jk = power(&Float { m: vec![j], e: 0 }, k, prec, work);
    let s = p as i64 - jk.e;
    if s < 0 {
        return Vec::new();
    }
    let mut q = Vec::new();
    quotient_into(&mut q, &[1], s, &jk.m, work);
    q
}
