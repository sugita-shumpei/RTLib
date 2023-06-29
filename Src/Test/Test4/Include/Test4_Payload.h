template<unsigned int I>
OTK_INLINE OTK_DEVICE void set_payload_uint1(unsigned int p)
{
	if constexpr (I == 0)
	{
		optixSetPayload_0(p);
	}
	if constexpr (I == 1)
	{
		optixSetPayload_1(p);
	}
	if constexpr (I == 2)
	{
		optixSetPayload_2(p);
	}
	if constexpr (I == 3)
	{
		optixSetPayload_3(p);
	}
	if constexpr (I == 4)
	{
		optixSetPayload_4(p);
	}
	if constexpr (I == 5)
	{
		optixSetPayload_5(p);
	}
	if constexpr (I == 6)
	{
		optixSetPayload_6(p);
	}
	if constexpr (I == 7)
	{
		optixSetPayload_7(p);
	}
	if constexpr (I == 8)
	{
		optixSetPayload_8(p);
	}
	if constexpr (I == 9)
	{
		optixSetPayload_9(p);
	}
	if constexpr (I == 10)
	{
		optixSetPayload_10(p);
	}
	if constexpr (I == 11)
	{
		optixSetPayload_11(p);
	}
	if constexpr (I == 12)
	{
		optixSetPayload_12(p);
	}
	if constexpr (I == 13)
	{
		optixSetPayload_13(p);
	}
	if constexpr (I == 14)
	{
		optixSetPayload_14(p);
	}
	if constexpr (I == 15)
	{
		optixSetPayload_15(p);
	}
	if constexpr (I == 16)
	{
		optixSetPayload_16(p);
	}
	if constexpr (I == 17)
	{
		optixSetPayload_17(p);
	}
	if constexpr (I == 18)
	{
		optixSetPayload_18(p);
	}
	if constexpr (I == 19)
	{
		optixSetPayload_19(p);
	}
	if constexpr (I == 20)
	{
		optixSetPayload_20(p);
	}
	if constexpr (I == 21)
	{
		optixSetPayload_21(p);
	}
	if constexpr (I == 22)
	{
		optixSetPayload_22(p);
	}
	if constexpr (I == 23)
	{
		optixSetPayload_23(p);
	}
	if constexpr (I == 24)
	{
		optixSetPayload_24(p);
	}
	if constexpr (I == 25)
	{
		optixSetPayload_25(p);
	}
	if constexpr (I == 26)
	{
		optixSetPayload_26(p);
	}
	if constexpr (I == 27)
	{
		optixSetPayload_27(p);
	}
	if constexpr (I == 28)
	{
		optixSetPayload_28(p);
	}
	if constexpr (I == 29)
	{
		optixSetPayload_29(p);
	}
	if constexpr (I == 30)
	{
		optixSetPayload_30(p);
	}
	if constexpr (I == 31)
	{
		optixSetPayload_31(p);
	}
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE auto get_payload_uint1() -> unsigned int
{
	if constexpr (I == 0)
	{
		return optixGetPayload_0();
	}
	if constexpr (I == 1)
	{
		return optixGetPayload_1();
	}
	if constexpr (I == 2)
	{
		return optixGetPayload_2();
	}
	if constexpr (I == 3)
	{
		return optixGetPayload_3();
	}
	if constexpr (I == 4)
	{
		return optixGetPayload_4();
	}
	if constexpr (I == 5)
	{
		return optixGetPayload_5();
	}
	if constexpr (I == 6)
	{
		return optixGetPayload_6();
	}
	if constexpr (I == 7)
	{
		return optixGetPayload_7();
	}
	if constexpr (I == 8)
	{
		return optixGetPayload_8();
	}
	if constexpr (I == 9)
	{
		return optixGetPayload_9();
	}
	if constexpr (I == 10)
	{
		return optixGetPayload_10();
	}
	if constexpr (I == 11)
	{
		return optixGetPayload_11();
	}
	if constexpr (I == 12)
	{
		return optixGetPayload_12();
	}
	if constexpr (I == 13)
	{
		return optixGetPayload_13();
	}
	if constexpr (I == 14)
	{
		return optixGetPayload_14();
	}
	if constexpr (I == 15)
	{
		return optixGetPayload_15();
	}
	if constexpr (I == 16)
	{
		return optixGetPayload_16();
	}
	if constexpr (I == 17)
	{
		return optixGetPayload_17();
	}
	if constexpr (I == 18)
	{
		return optixGetPayload_18();
	}
	if constexpr (I == 19)
	{
		return optixGetPayload_19();
	}
	if constexpr (I == 20)
	{
		return optixGetPayload_20();
	}
	if constexpr (I == 21)
	{
		return optixGetPayload_21();
	}
	if constexpr (I == 22)
	{
		return optixGetPayload_22();
	}
	if constexpr (I == 23)
	{
		return optixGetPayload_23();
	}
	if constexpr (I == 24)
	{
		return optixGetPayload_24();
	}
	if constexpr (I == 25)
	{
		return optixGetPayload_25();
	}
	if constexpr (I == 26)
	{
		return optixGetPayload_26();
	}
	if constexpr (I == 27)
	{
		return optixGetPayload_27();
	}
	if constexpr (I == 28)
	{
		return optixGetPayload_28();
	}
	if constexpr (I == 29)
	{
		return optixGetPayload_29();
	}
	if constexpr (I == 30)
	{
		return optixGetPayload_30();
	}
	if constexpr (I == 31)
	{
		return optixGetPayload_31();
	}
	return {};
}
template<unsigned int I>
OTK_INLINE OTK_DEVICE void set_payload_uint2(uint2 p)
{
	if constexpr (I == 0)
	{
		optixSetPayload_0(p.x);
		optixSetPayload_1(p.y);
	}
	if constexpr (I == 1)
	{
		optixSetPayload_1(p.x);
		optixSetPayload_2(p.y);
	}
	if constexpr (I == 2)
	{
		optixSetPayload_2(p.x);
		optixSetPayload_3(p.y);
	}
	if constexpr (I == 3)
	{
		optixSetPayload_3(p.x);
		optixSetPayload_4(p.y);
	}
	if constexpr (I == 4)
	{
		optixSetPayload_4(p.x);
		optixSetPayload_5(p.y);
	}
	if constexpr (I == 5)
	{
		optixSetPayload_5(p.x);
		optixSetPayload_6(p.y);
	}
	if constexpr (I == 6)
	{
		optixSetPayload_6(p.x);
		optixSetPayload_7(p.y);
	}
	if constexpr (I == 7)
	{
		optixSetPayload_7(p.x);
		optixSetPayload_8(p.y);
	}
	if constexpr (I == 8)
	{
		optixSetPayload_8(p.x);
		optixSetPayload_9(p.y);
	}
	if constexpr (I == 9)
	{
		optixSetPayload_9(p.x);
		optixSetPayload_10(p.y);
	}
	if constexpr (I == 10)
	{
		optixSetPayload_10(p.x);
		optixSetPayload_11(p.y);
	}
	if constexpr (I == 11)
	{
		optixSetPayload_11(p.x);
		optixSetPayload_12(p.y);
	}
	if constexpr (I == 12)
	{
		optixSetPayload_12(p.x);
		optixSetPayload_13(p.y);
	}
	if constexpr (I == 13)
	{
		optixSetPayload_13(p.x);
		optixSetPayload_14(p.y);
	}
	if constexpr (I == 14)
	{
		optixSetPayload_14(p.x);
		optixSetPayload_15(p.y);
	}
	if constexpr (I == 15)
	{
		optixSetPayload_15(p.x);
		optixSetPayload_16(p.y);
	}
	if constexpr (I == 16)
	{
		optixSetPayload_16(p.x);
		optixSetPayload_17(p.y);
	}
	if constexpr (I == 17)
	{
		optixSetPayload_17(p.x);
		optixSetPayload_18(p.y);
	}
	if constexpr (I == 18)
	{
		optixSetPayload_18(p.x);
		optixSetPayload_19(p.y);
	}
	if constexpr (I == 19)
	{
		optixSetPayload_19(p.x);
		optixSetPayload_20(p.y);
	}
	if constexpr (I == 20)
	{
		optixSetPayload_20(p.x);
		optixSetPayload_21(p.y);
	}
	if constexpr (I == 21)
	{
		optixSetPayload_21(p.x);
		optixSetPayload_22(p.y);
	}
	if constexpr (I == 22)
	{
		optixSetPayload_22(p.x);
		optixSetPayload_23(p.y);
	}
	if constexpr (I == 23)
	{
		optixSetPayload_23(p.x);
		optixSetPayload_24(p.y);
	}
	if constexpr (I == 24)
	{
		optixSetPayload_24(p.x);
		optixSetPayload_25(p.y);
	}
	if constexpr (I == 25)
	{
		optixSetPayload_25(p.x);
		optixSetPayload_26(p.y);
	}
	if constexpr (I == 26)
	{
		optixSetPayload_26(p.x);
		optixSetPayload_27(p.y);
	}
	if constexpr (I == 27)
	{
		optixSetPayload_27(p.x);
		optixSetPayload_28(p.y);
	}
	if constexpr (I == 28)
	{
		optixSetPayload_28(p.x);
		optixSetPayload_29(p.y);
	}
	if constexpr (I == 29)
	{
		optixSetPayload_29(p.x);
		optixSetPayload_30(p.y);
	}
	if constexpr (I == 30)
	{
		optixSetPayload_30(p.x);
		optixSetPayload_31(p.y);
	}
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE auto get_payload_uint2() -> uint2
{
	if constexpr (I == 0)
	{
		return { optixGetPayload_0(),optixGetPayload_1() };
	}
	if constexpr (I == 1)
	{
		return { optixGetPayload_1(),optixGetPayload_2() };
	}
	if constexpr (I == 2)
	{
		return { optixGetPayload_2(),optixGetPayload_3() };
	}
	if constexpr (I == 3)
	{
		return { optixGetPayload_3(),optixGetPayload_4() };
	}
	if constexpr (I == 4)
	{
		return { optixGetPayload_4(),optixGetPayload_5() };
	}
	if constexpr (I == 5)
	{
		return { optixGetPayload_5(),optixGetPayload_6() };
	}
	if constexpr (I == 6)
	{
		return { optixGetPayload_6(),optixGetPayload_7() };
	}
	if constexpr (I == 7)
	{
		return { optixGetPayload_7(),optixGetPayload_8() };
	}
	if constexpr (I == 8)
	{
		return { optixGetPayload_8(),optixGetPayload_9() };
	}
	if constexpr (I == 9)
	{
		return { optixGetPayload_9(),optixGetPayload_10() };
	}
	if constexpr (I == 10)
	{
		return { optixGetPayload_10(),optixGetPayload_11() };
	}
	if constexpr (I == 11)
	{
		return { optixGetPayload_11(),optixGetPayload_12() };
	}
	if constexpr (I == 12)
	{
		return { optixGetPayload_12(),optixGetPayload_13() };
	}
	if constexpr (I == 13)
	{
		return { optixGetPayload_13(),optixGetPayload_14() };
	}
	if constexpr (I == 14)
	{
		return { optixGetPayload_14(),optixGetPayload_15() };
	}
	if constexpr (I == 15)
	{
		return { optixGetPayload_15(),optixGetPayload_16() };
	}
	if constexpr (I == 16)
	{
		return { optixGetPayload_16(),optixGetPayload_17() };
	}
	if constexpr (I == 17)
	{
		return { optixGetPayload_17(),optixGetPayload_18() };
	}
	if constexpr (I == 18)
	{
		return { optixGetPayload_18(),optixGetPayload_19() };
	}
	if constexpr (I == 19)
	{
		return { optixGetPayload_19(),optixGetPayload_20() };
	}
	if constexpr (I == 20)
	{
		return { optixGetPayload_20(),optixGetPayload_21() };
	}
	if constexpr (I == 21)
	{
		return { optixGetPayload_21(),optixGetPayload_22() };
	}
	if constexpr (I == 22)
	{
		return { optixGetPayload_22(),optixGetPayload_23() };
	}
	if constexpr (I == 23)
	{
		return { optixGetPayload_23(),optixGetPayload_24() };
	}
	if constexpr (I == 24)
	{
		return { optixGetPayload_24(),optixGetPayload_25() };
	}
	if constexpr (I == 25)
	{
		return { optixGetPayload_25(),optixGetPayload_26() };
	}
	if constexpr (I == 26)
	{
		return { optixGetPayload_26(),optixGetPayload_27() };
	}
	if constexpr (I == 27)
	{
		return { optixGetPayload_27(),optixGetPayload_28() };
	}
	if constexpr (I == 28)
	{
		return { optixGetPayload_28(),optixGetPayload_29() };
	}
	if constexpr (I == 29)
	{
		return { optixGetPayload_29(),optixGetPayload_30() };
	}
	if constexpr (I == 30)
	{
		return { optixGetPayload_30(),optixGetPayload_31() };
	}
	return {};
}
template<unsigned int I>
OTK_INLINE OTK_DEVICE void set_payload_uint3(uint3 p)
{
	if constexpr (I == 0)
	{
		optixSetPayload_0(p.x);
		optixSetPayload_1(p.y);
		optixSetPayload_2(p.z);
	}
	if constexpr (I == 1)
	{
		optixSetPayload_1(p.x);
		optixSetPayload_2(p.y);
		optixSetPayload_3(p.z);
	}
	if constexpr (I == 2)
	{
		optixSetPayload_2(p.x);
		optixSetPayload_3(p.y);
		optixSetPayload_4(p.z);
	}
	if constexpr (I == 3)
	{
		optixSetPayload_3(p.x);
		optixSetPayload_4(p.y);
		optixSetPayload_5(p.z);
	}
	if constexpr (I == 4)
	{
		optixSetPayload_4(p.x);
		optixSetPayload_5(p.y);
		optixSetPayload_6(p.z);
	}
	if constexpr (I == 5)
	{
		optixSetPayload_5(p.x);
		optixSetPayload_6(p.y);
		optixSetPayload_7(p.z);
	}
	if constexpr (I == 6)
	{
		optixSetPayload_6(p.x);
		optixSetPayload_7(p.y);
		optixSetPayload_8(p.z);
	}
	if constexpr (I == 7)
	{
		optixSetPayload_7(p.x);
		optixSetPayload_8(p.y);
		optixSetPayload_9(p.z);
	}
	if constexpr (I == 8)
	{
		optixSetPayload_8(p.x);
		optixSetPayload_9(p.y);
		optixSetPayload_10(p.z);
	}
	if constexpr (I == 9)
	{
		optixSetPayload_9(p.x);
		optixSetPayload_10(p.y);
		optixSetPayload_11(p.z);
	}
	if constexpr (I == 10)
	{
		optixSetPayload_10(p.x);
		optixSetPayload_11(p.y);
		optixSetPayload_12(p.z);
	}
	if constexpr (I == 11)
	{
		optixSetPayload_11(p.x);
		optixSetPayload_12(p.y);
		optixSetPayload_13(p.z);
	}
	if constexpr (I == 12)
	{
		optixSetPayload_12(p.x);
		optixSetPayload_13(p.y);
		optixSetPayload_14(p.z);
	}
	if constexpr (I == 13)
	{
		optixSetPayload_13(p.x);
		optixSetPayload_14(p.y);
		optixSetPayload_15(p.z);
	}
	if constexpr (I == 14)
	{
		optixSetPayload_14(p.x);
		optixSetPayload_15(p.y);
		optixSetPayload_16(p.z);
	}
	if constexpr (I == 15)
	{
		optixSetPayload_15(p.x);
		optixSetPayload_16(p.y);
		optixSetPayload_17(p.z);
	}
	if constexpr (I == 16)
	{
		optixSetPayload_16(p.x);
		optixSetPayload_17(p.y);
		optixSetPayload_18(p.z);
	}
	if constexpr (I == 17)
	{
		optixSetPayload_17(p.x);
		optixSetPayload_18(p.y);
		optixSetPayload_19(p.z);
	}
	if constexpr (I == 18)
	{
		optixSetPayload_18(p.x);
		optixSetPayload_19(p.y);
		optixSetPayload_20(p.z);
	}
	if constexpr (I == 19)
	{
		optixSetPayload_19(p.x);
		optixSetPayload_20(p.y);
		optixSetPayload_21(p.z);
	}
	if constexpr (I == 20)
	{
		optixSetPayload_20(p.x);
		optixSetPayload_21(p.y);
		optixSetPayload_22(p.z);
	}
	if constexpr (I == 21)
	{
		optixSetPayload_21(p.x);
		optixSetPayload_22(p.y);
		optixSetPayload_23(p.z);
	}
	if constexpr (I == 22)
	{
		optixSetPayload_22(p.x);
		optixSetPayload_23(p.y);
		optixSetPayload_24(p.z);
	}
	if constexpr (I == 23)
	{
		optixSetPayload_23(p.x);
		optixSetPayload_24(p.y);
		optixSetPayload_25(p.z);
	}
	if constexpr (I == 24)
	{
		optixSetPayload_24(p.x);
		optixSetPayload_25(p.y);
		optixSetPayload_26(p.z);
	}
	if constexpr (I == 25)
	{
		optixSetPayload_25(p.x);
		optixSetPayload_26(p.y);
		optixSetPayload_27(p.z);
	}
	if constexpr (I == 26)
	{
		optixSetPayload_26(p.x);
		optixSetPayload_27(p.y);
		optixSetPayload_28(p.z);
	}
	if constexpr (I == 27)
	{
		optixSetPayload_27(p.x);
		optixSetPayload_28(p.y);
		optixSetPayload_29(p.z);
	}
	if constexpr (I == 28)
	{
		optixSetPayload_28(p.x);
		optixSetPayload_29(p.y);
		optixSetPayload_30(p.z);
	}
	if constexpr (I == 29)
	{
		optixSetPayload_29(p.x);
		optixSetPayload_30(p.y);
		optixSetPayload_31(p.z);
	}
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE auto get_payload_uint3() -> uint3
{
	if constexpr (I == 0)
	{
		return { optixGetPayload_0(),optixGetPayload_1(),optixGetPayload_2() };
	}
	if constexpr (I == 1)
	{
		return { optixGetPayload_1(),optixGetPayload_2(),optixGetPayload_3() };
	}
	if constexpr (I == 2)
	{
		return { optixGetPayload_2(),optixGetPayload_3(),optixGetPayload_4() };
	}
	if constexpr (I == 3)
	{
		return { optixGetPayload_3(),optixGetPayload_4(),optixGetPayload_5() };
	}
	if constexpr (I == 4)
	{
		return { optixGetPayload_4(),optixGetPayload_5(),optixGetPayload_6() };
	}
	if constexpr (I == 5)
	{
		return { optixGetPayload_5(),optixGetPayload_6(),optixGetPayload_7() };
	}
	if constexpr (I == 6)
	{
		return { optixGetPayload_6(),optixGetPayload_7(),optixGetPayload_8() };
	}
	if constexpr (I == 7)
	{
		return { optixGetPayload_7(),optixGetPayload_8(),optixGetPayload_9() };
	}
	if constexpr (I == 8)
	{
		return { optixGetPayload_8(),optixGetPayload_9(),optixGetPayload_10() };
	}
	if constexpr (I == 9)
	{
		return { optixGetPayload_9(),optixGetPayload_10(),optixGetPayload_11() };
	}
	if constexpr (I == 10)
	{
		return { optixGetPayload_10(),optixGetPayload_11(),optixGetPayload_12() };
	}
	if constexpr (I == 11)
	{
		return { optixGetPayload_11(),optixGetPayload_12(),optixGetPayload_13() };
	}
	if constexpr (I == 12)
	{
		return { optixGetPayload_12(),optixGetPayload_13(),optixGetPayload_14() };
	}
	if constexpr (I == 13)
	{
		return { optixGetPayload_13(),optixGetPayload_14(),optixGetPayload_15() };
	}
	if constexpr (I == 14)
	{
		return { optixGetPayload_14(),optixGetPayload_15(),optixGetPayload_16() };
	}
	if constexpr (I == 15)
	{
		return { optixGetPayload_15(),optixGetPayload_16(),optixGetPayload_17() };
	}
	if constexpr (I == 16)
	{
		return { optixGetPayload_16(),optixGetPayload_17(),optixGetPayload_18() };
	}
	if constexpr (I == 17)
	{
		return { optixGetPayload_17(),optixGetPayload_18(),optixGetPayload_19() };
	}
	if constexpr (I == 18)
	{
		return { optixGetPayload_18(),optixGetPayload_19(),optixGetPayload_20() };
	}
	if constexpr (I == 19)
	{
		return { optixGetPayload_19(),optixGetPayload_20(),optixGetPayload_21() };
	}
	if constexpr (I == 20)
	{
		return { optixGetPayload_20(),optixGetPayload_21(),optixGetPayload_22() };
	}
	if constexpr (I == 21)
	{
		return { optixGetPayload_21(),optixGetPayload_22(),optixGetPayload_23() };
	}
	if constexpr (I == 22)
	{
		return { optixGetPayload_22(),optixGetPayload_23(),optixGetPayload_24() };
	}
	if constexpr (I == 23)
	{
		return { optixGetPayload_23(),optixGetPayload_24(),optixGetPayload_25() };
	}
	if constexpr (I == 24)
	{
		return { optixGetPayload_24(),optixGetPayload_25(),optixGetPayload_26() };
	}
	if constexpr (I == 25)
	{
		return { optixGetPayload_25(),optixGetPayload_26(),optixGetPayload_27() };
	}
	if constexpr (I == 26)
	{
		return { optixGetPayload_26(),optixGetPayload_27(),optixGetPayload_28() };
	}
	if constexpr (I == 27)
	{
		return { optixGetPayload_27(),optixGetPayload_28(),optixGetPayload_29() };
	}
	if constexpr (I == 28)
	{
		return { optixGetPayload_28(),optixGetPayload_29(),optixGetPayload_30() };
	}
	if constexpr (I == 29)
	{
		return { optixGetPayload_29(),optixGetPayload_30(),optixGetPayload_31() };
	}
	return {};
}
template<unsigned int I>
OTK_INLINE OTK_DEVICE void set_payload_uint4(uint4 p)
{
	if constexpr (I == 0)
	{
		optixSetPayload_0(p.x);
		optixSetPayload_1(p.y);
		optixSetPayload_2(p.z);
		optixSetPayload_3(p.w);
	}
	if constexpr (I == 1)
	{
		optixSetPayload_1(p.x);
		optixSetPayload_2(p.y);
		optixSetPayload_3(p.z);
		optixSetPayload_4(p.w);
	}
	if constexpr (I == 2)
	{
		optixSetPayload_2(p.x);
		optixSetPayload_3(p.y);
		optixSetPayload_4(p.z);
		optixSetPayload_5(p.w);
	}
	if constexpr (I == 3)
	{
		optixSetPayload_3(p.x);
		optixSetPayload_4(p.y);
		optixSetPayload_5(p.z);
		optixSetPayload_6(p.w);
	}
	if constexpr (I == 4)
	{
		optixSetPayload_4(p.x);
		optixSetPayload_5(p.y);
		optixSetPayload_6(p.z);
		optixSetPayload_7(p.w);
	}
	if constexpr (I == 5)
	{
		optixSetPayload_5(p.x);
		optixSetPayload_6(p.y);
		optixSetPayload_7(p.z);
		optixSetPayload_8(p.w);
	}
	if constexpr (I == 6)
	{
		optixSetPayload_6(p.x);
		optixSetPayload_7(p.y);
		optixSetPayload_8(p.z);
		optixSetPayload_9(p.w);
	}
	if constexpr (I == 7)
	{
		optixSetPayload_7(p.x);
		optixSetPayload_8(p.y);
		optixSetPayload_9(p.z);
		optixSetPayload_10(p.w);
	}
	if constexpr (I == 8)
	{
		optixSetPayload_8(p.x);
		optixSetPayload_9(p.y);
		optixSetPayload_10(p.z);
		optixSetPayload_11(p.w);
	}
	if constexpr (I == 9)
	{
		optixSetPayload_9(p.x);
		optixSetPayload_10(p.y);
		optixSetPayload_11(p.z);
		optixSetPayload_12(p.w);
	}
	if constexpr (I == 10)
	{
		optixSetPayload_10(p.x);
		optixSetPayload_11(p.y);
		optixSetPayload_12(p.z);
		optixSetPayload_13(p.w);
	}
	if constexpr (I == 11)
	{
		optixSetPayload_11(p.x);
		optixSetPayload_12(p.y);
		optixSetPayload_13(p.z);
		optixSetPayload_14(p.w);
	}
	if constexpr (I == 12)
	{
		optixSetPayload_12(p.x);
		optixSetPayload_13(p.y);
		optixSetPayload_14(p.z);
		optixSetPayload_15(p.w);
	}
	if constexpr (I == 13)
	{
		optixSetPayload_13(p.x);
		optixSetPayload_14(p.y);
		optixSetPayload_15(p.z);
		optixSetPayload_16(p.w);
	}
	if constexpr (I == 14)
	{
		optixSetPayload_14(p.x);
		optixSetPayload_15(p.y);
		optixSetPayload_16(p.z);
		optixSetPayload_17(p.w);
	}
	if constexpr (I == 15)
	{
		optixSetPayload_15(p.x);
		optixSetPayload_16(p.y);
		optixSetPayload_17(p.z);
		optixSetPayload_18(p.w);
	}
	if constexpr (I == 16)
	{
		optixSetPayload_16(p.x);
		optixSetPayload_17(p.y);
		optixSetPayload_18(p.z);
		optixSetPayload_19(p.w);
	}
	if constexpr (I == 17)
	{
		optixSetPayload_17(p.x);
		optixSetPayload_18(p.y);
		optixSetPayload_19(p.z);
		optixSetPayload_20(p.w);
	}
	if constexpr (I == 18)
	{
		optixSetPayload_18(p.x);
		optixSetPayload_19(p.y);
		optixSetPayload_20(p.z);
		optixSetPayload_21(p.w);
	}
	if constexpr (I == 19)
	{
		optixSetPayload_19(p.x);
		optixSetPayload_20(p.y);
		optixSetPayload_21(p.z);
		optixSetPayload_22(p.w);
	}
	if constexpr (I == 20)
	{
		optixSetPayload_20(p.x);
		optixSetPayload_21(p.y);
		optixSetPayload_22(p.z);
		optixSetPayload_23(p.w);
	}
	if constexpr (I == 21)
	{
		optixSetPayload_21(p.x);
		optixSetPayload_22(p.y);
		optixSetPayload_23(p.z);
		optixSetPayload_24(p.w);
	}
	if constexpr (I == 22)
	{
		optixSetPayload_22(p.x);
		optixSetPayload_23(p.y);
		optixSetPayload_24(p.z);
		optixSetPayload_25(p.w);
	}
	if constexpr (I == 23)
	{
		optixSetPayload_23(p.x);
		optixSetPayload_24(p.y);
		optixSetPayload_25(p.z);
		optixSetPayload_26(p.w);
	}
	if constexpr (I == 24)
	{
		optixSetPayload_24(p.x);
		optixSetPayload_25(p.y);
		optixSetPayload_26(p.z);
		optixSetPayload_27(p.w);
	}
	if constexpr (I == 25)
	{
		optixSetPayload_25(p.x);
		optixSetPayload_26(p.y);
		optixSetPayload_27(p.z);
		optixSetPayload_28(p.w);
	}
	if constexpr (I == 26)
	{
		optixSetPayload_26(p.x);
		optixSetPayload_27(p.y);
		optixSetPayload_28(p.z);
		optixSetPayload_29(p.w);
	}
	if constexpr (I == 27)
	{
		optixSetPayload_27(p.x);
		optixSetPayload_28(p.y);
		optixSetPayload_29(p.z);
		optixSetPayload_30(p.w);
	}
	if constexpr (I == 28)
	{
		optixSetPayload_28(p.x);
		optixSetPayload_29(p.y);
		optixSetPayload_30(p.z);
		optixSetPayload_31(p.w);
	}
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE auto get_payload_uint4() -> uint4
{
	if constexpr (I == 0)
	{
		return { optixGetPayload_0(),optixGetPayload_1(),optixGetPayload_2(),optixGetPayload_3() };
	}
	if constexpr (I == 1)
	{
		return { optixGetPayload_1(),optixGetPayload_2(),optixGetPayload_3(),optixGetPayload_4() };
	}
	if constexpr (I == 2)
	{
		return { optixGetPayload_2(),optixGetPayload_3(),optixGetPayload_4(),optixGetPayload_5() };
	}
	if constexpr (I == 3)
	{
		return { optixGetPayload_3(),optixGetPayload_4(),optixGetPayload_5(),optixGetPayload_6() };
	}
	if constexpr (I == 4)
	{
		return { optixGetPayload_4(),optixGetPayload_5(),optixGetPayload_6(),optixGetPayload_7() };
	}
	if constexpr (I == 5)
	{
		return { optixGetPayload_5(),optixGetPayload_6(),optixGetPayload_7(),optixGetPayload_8() };
	}
	if constexpr (I == 6)
	{
		return { optixGetPayload_6(),optixGetPayload_7(),optixGetPayload_8(),optixGetPayload_9() };
	}
	if constexpr (I == 7)
	{
		return { optixGetPayload_7(),optixGetPayload_8(),optixGetPayload_9(),optixGetPayload_10() };
	}
	if constexpr (I == 8)
	{
		return { optixGetPayload_8(),optixGetPayload_9(),optixGetPayload_10(),optixGetPayload_11() };
	}
	if constexpr (I == 9)
	{
		return { optixGetPayload_9(),optixGetPayload_10(),optixGetPayload_11(),optixGetPayload_12() };
	}
	if constexpr (I == 10)
	{
		return { optixGetPayload_10(),optixGetPayload_11(),optixGetPayload_12(),optixGetPayload_13() };
	}
	if constexpr (I == 11)
	{
		return { optixGetPayload_11(),optixGetPayload_12(),optixGetPayload_13(),optixGetPayload_14() };
	}
	if constexpr (I == 12)
	{
		return { optixGetPayload_12(),optixGetPayload_13(),optixGetPayload_14(),optixGetPayload_15() };
	}
	if constexpr (I == 13)
	{
		return { optixGetPayload_13(),optixGetPayload_14(),optixGetPayload_15(),optixGetPayload_16() };
	}
	if constexpr (I == 14)
	{
		return { optixGetPayload_14(),optixGetPayload_15(),optixGetPayload_16(),optixGetPayload_17() };
	}
	if constexpr (I == 15)
	{
		return { optixGetPayload_15(),optixGetPayload_16(),optixGetPayload_17(),optixGetPayload_18() };
	}
	if constexpr (I == 16)
	{
		return { optixGetPayload_16(),optixGetPayload_17(),optixGetPayload_18(),optixGetPayload_19() };
	}
	if constexpr (I == 17)
	{
		return { optixGetPayload_17(),optixGetPayload_18(),optixGetPayload_19(),optixGetPayload_20() };
	}
	if constexpr (I == 18)
	{
		return { optixGetPayload_18(),optixGetPayload_19(),optixGetPayload_20(),optixGetPayload_21() };
	}
	if constexpr (I == 19)
	{
		return { optixGetPayload_19(),optixGetPayload_20(),optixGetPayload_21(),optixGetPayload_22() };
	}
	if constexpr (I == 20)
	{
		return { optixGetPayload_20(),optixGetPayload_21(),optixGetPayload_22(),optixGetPayload_23() };
	}
	if constexpr (I == 21)
	{
		return { optixGetPayload_21(),optixGetPayload_22(),optixGetPayload_23(),optixGetPayload_24() };
	}
	if constexpr (I == 22)
	{
		return { optixGetPayload_22(),optixGetPayload_23(),optixGetPayload_24(),optixGetPayload_25() };
	}
	if constexpr (I == 23)
	{
		return { optixGetPayload_23(),optixGetPayload_24(),optixGetPayload_25(),optixGetPayload_26() };
	}
	if constexpr (I == 24)
	{
		return { optixGetPayload_24(),optixGetPayload_25(),optixGetPayload_26(),optixGetPayload_27() };
	}
	if constexpr (I == 25)
	{
		return { optixGetPayload_25(),optixGetPayload_26(),optixGetPayload_27(),optixGetPayload_28() };
	}
	if constexpr (I == 26)
	{
		return { optixGetPayload_26(),optixGetPayload_27(),optixGetPayload_28(),optixGetPayload_29() };
	}
	if constexpr (I == 27)
	{
		return { optixGetPayload_27(),optixGetPayload_28(),optixGetPayload_29(),optixGetPayload_30() };
	}
	if constexpr (I == 28)
	{
		return { optixGetPayload_28(),optixGetPayload_29(),optixGetPayload_30(),optixGetPayload_31() };
	}
	return {};
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE void set_payload_float1(float p)
{
	if constexpr (I == 0)
	{
		optixSetPayload_0(__float_as_uint(p));
	}
	if constexpr (I == 1)
	{
		optixSetPayload_1(__float_as_uint(p));
	}
	if constexpr (I == 2)
	{
		optixSetPayload_2(__float_as_uint(p));
	}
	if constexpr (I == 3)
	{
		optixSetPayload_3(__float_as_uint(p));
	}
	if constexpr (I == 4)
	{
		optixSetPayload_4(__float_as_uint(p));
	}
	if constexpr (I == 5)
	{
		optixSetPayload_5(__float_as_uint(p));
	}
	if constexpr (I == 6)
	{
		optixSetPayload_6(__float_as_uint(p));
	}
	if constexpr (I == 7)
	{
		optixSetPayload_7(__float_as_uint(p));
	}
	if constexpr (I == 8)
	{
		optixSetPayload_8(__float_as_uint(p));
	}
	if constexpr (I == 9)
	{
		optixSetPayload_9(__float_as_uint(p));
	}
	if constexpr (I == 10)
	{
		optixSetPayload_10(__float_as_uint(p));
	}
	if constexpr (I == 11)
	{
		optixSetPayload_11(__float_as_uint(p));
	}
	if constexpr (I == 12)
	{
		optixSetPayload_12(__float_as_uint(p));
	}
	if constexpr (I == 13)
	{
		optixSetPayload_13(__float_as_uint(p));
	}
	if constexpr (I == 14)
	{
		optixSetPayload_14(__float_as_uint(p));
	}
	if constexpr (I == 15)
	{
		optixSetPayload_15(__float_as_uint(p));
	}
	if constexpr (I == 16)
	{
		optixSetPayload_16(__float_as_uint(p));
	}
	if constexpr (I == 17)
	{
		optixSetPayload_17(__float_as_uint(p));
	}
	if constexpr (I == 18)
	{
		optixSetPayload_18(__float_as_uint(p));
	}
	if constexpr (I == 19)
	{
		optixSetPayload_19(__float_as_uint(p));
	}
	if constexpr (I == 20)
	{
		optixSetPayload_20(__float_as_uint(p));
	}
	if constexpr (I == 21)
	{
		optixSetPayload_21(__float_as_uint(p));
	}
	if constexpr (I == 22)
	{
		optixSetPayload_22(__float_as_uint(p));
	}
	if constexpr (I == 23)
	{
		optixSetPayload_23(__float_as_uint(p));
	}
	if constexpr (I == 24)
	{
		optixSetPayload_24(__float_as_uint(p));
	}
	if constexpr (I == 25)
	{
		optixSetPayload_25(__float_as_uint(p));
	}
	if constexpr (I == 26)
	{
		optixSetPayload_26(__float_as_uint(p));
	}
	if constexpr (I == 27)
	{
		optixSetPayload_27(__float_as_uint(p));
	}
	if constexpr (I == 28)
	{
		optixSetPayload_28(__float_as_uint(p));
	}
	if constexpr (I == 29)
	{
		optixSetPayload_29(__float_as_uint(p));
	}
	if constexpr (I == 30)
	{
		optixSetPayload_30(__float_as_uint(p));
	}
	if constexpr (I == 31)
	{
		optixSetPayload_31(__float_as_uint(p));
	}
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE auto get_payload_float1() -> float
{
	if constexpr (I == 0)
	{
		return __uint_as_float(optixGetPayload_0());
	}
	if constexpr (I == 1)
	{
		return __uint_as_float(optixGetPayload_1());
	}
	if constexpr (I == 2)
	{
		return __uint_as_float(optixGetPayload_2());
	}
	if constexpr (I == 3)
	{
		return __uint_as_float(optixGetPayload_3());
	}
	if constexpr (I == 4)
	{
		return __uint_as_float(optixGetPayload_4());
	}
	if constexpr (I == 5)
	{
		return __uint_as_float(optixGetPayload_5());
	}
	if constexpr (I == 6)
	{
		return __uint_as_float(optixGetPayload_6());
	}
	if constexpr (I == 7)
	{
		return __uint_as_float(optixGetPayload_7());
	}
	if constexpr (I == 8)
	{
		return __uint_as_float(optixGetPayload_8());
	}
	if constexpr (I == 9)
	{
		return __uint_as_float(optixGetPayload_9());
	}
	if constexpr (I == 10)
	{
		return __uint_as_float(optixGetPayload_10());
	}
	if constexpr (I == 11)
	{
		return __uint_as_float(optixGetPayload_11());
	}
	if constexpr (I == 12)
	{
		return __uint_as_float(optixGetPayload_12());
	}
	if constexpr (I == 13)
	{
		return __uint_as_float(optixGetPayload_13());
	}
	if constexpr (I == 14)
	{
		return __uint_as_float(optixGetPayload_14());
	}
	if constexpr (I == 15)
	{
		return __uint_as_float(optixGetPayload_15());
	}
	if constexpr (I == 16)
	{
		return __uint_as_float(optixGetPayload_16());
	}
	if constexpr (I == 17)
	{
		return __uint_as_float(optixGetPayload_17());
	}
	if constexpr (I == 18)
	{
		return __uint_as_float(optixGetPayload_18());
	}
	if constexpr (I == 19)
	{
		return __uint_as_float(optixGetPayload_19());
	}
	if constexpr (I == 20)
	{
		return __uint_as_float(optixGetPayload_20());
	}
	if constexpr (I == 21)
	{
		return __uint_as_float(optixGetPayload_21());
	}
	if constexpr (I == 22)
	{
		return __uint_as_float(optixGetPayload_22());
	}
	if constexpr (I == 23)
	{
		return __uint_as_float(optixGetPayload_23());
	}
	if constexpr (I == 24)
	{
		return __uint_as_float(optixGetPayload_24());
	}
	if constexpr (I == 25)
	{
		return __uint_as_float(optixGetPayload_25());
	}
	if constexpr (I == 26)
	{
		return __uint_as_float(optixGetPayload_26());
	}
	if constexpr (I == 27)
	{
		return __uint_as_float(optixGetPayload_27());
	}
	if constexpr (I == 28)
	{
		return __uint_as_float(optixGetPayload_28());
	}
	if constexpr (I == 29)
	{
		return __uint_as_float(optixGetPayload_29());
	}
	if constexpr (I == 30)
	{
		return __uint_as_float(optixGetPayload_30());
	}
	if constexpr (I == 31)
	{
		return __uint_as_float(optixGetPayload_31());
	}
	return {};
}
template<unsigned int I>
OTK_INLINE OTK_DEVICE void set_payload_float2(float2 p)
{
	if constexpr (I == 0)
	{
		optixSetPayload_0(__float_as_uint(p.x));
		optixSetPayload_1(__float_as_uint(p.y));
	}
	if constexpr (I == 1)
	{
		optixSetPayload_1(__float_as_uint(p.x));
		optixSetPayload_2(__float_as_uint(p.y));
	}
	if constexpr (I == 2)
	{
		optixSetPayload_2(__float_as_uint(p.x));
		optixSetPayload_3(__float_as_uint(p.y));
	}
	if constexpr (I == 3)
	{
		optixSetPayload_3(__float_as_uint(p.x));
		optixSetPayload_4(__float_as_uint(p.y));
	}
	if constexpr (I == 4)
	{
		optixSetPayload_4(__float_as_uint(p.x));
		optixSetPayload_5(__float_as_uint(p.y));
	}
	if constexpr (I == 5)
	{
		optixSetPayload_5(__float_as_uint(p.x));
		optixSetPayload_6(__float_as_uint(p.y));
	}
	if constexpr (I == 6)
	{
		optixSetPayload_6(__float_as_uint(p.x));
		optixSetPayload_7(__float_as_uint(p.y));
	}
	if constexpr (I == 7)
	{
		optixSetPayload_7(__float_as_uint(p.x));
		optixSetPayload_8(__float_as_uint(p.y));
	}
	if constexpr (I == 8)
	{
		optixSetPayload_8(__float_as_uint(p.x));
		optixSetPayload_9(__float_as_uint(p.y));
	}
	if constexpr (I == 9)
	{
		optixSetPayload_9(__float_as_uint(p.x));
		optixSetPayload_10(__float_as_uint(p.y));
	}
	if constexpr (I == 10)
	{
		optixSetPayload_10(__float_as_uint(p.x));
		optixSetPayload_11(__float_as_uint(p.y));
	}
	if constexpr (I == 11)
	{
		optixSetPayload_11(__float_as_uint(p.x));
		optixSetPayload_12(__float_as_uint(p.y));
	}
	if constexpr (I == 12)
	{
		optixSetPayload_12(__float_as_uint(p.x));
		optixSetPayload_13(__float_as_uint(p.y));
	}
	if constexpr (I == 13)
	{
		optixSetPayload_13(__float_as_uint(p.x));
		optixSetPayload_14(__float_as_uint(p.y));
	}
	if constexpr (I == 14)
	{
		optixSetPayload_14(__float_as_uint(p.x));
		optixSetPayload_15(__float_as_uint(p.y));
	}
	if constexpr (I == 15)
	{
		optixSetPayload_15(__float_as_uint(p.x));
		optixSetPayload_16(__float_as_uint(p.y));
	}
	if constexpr (I == 16)
	{
		optixSetPayload_16(__float_as_uint(p.x));
		optixSetPayload_17(__float_as_uint(p.y));
	}
	if constexpr (I == 17)
	{
		optixSetPayload_17(__float_as_uint(p.x));
		optixSetPayload_18(__float_as_uint(p.y));
	}
	if constexpr (I == 18)
	{
		optixSetPayload_18(__float_as_uint(p.x));
		optixSetPayload_19(__float_as_uint(p.y));
	}
	if constexpr (I == 19)
	{
		optixSetPayload_19(__float_as_uint(p.x));
		optixSetPayload_20(__float_as_uint(p.y));
	}
	if constexpr (I == 20)
	{
		optixSetPayload_20(__float_as_uint(p.x));
		optixSetPayload_21(__float_as_uint(p.y));
	}
	if constexpr (I == 21)
	{
		optixSetPayload_21(__float_as_uint(p.x));
		optixSetPayload_22(__float_as_uint(p.y));
	}
	if constexpr (I == 22)
	{
		optixSetPayload_22(__float_as_uint(p.x));
		optixSetPayload_23(__float_as_uint(p.y));
	}
	if constexpr (I == 23)
	{
		optixSetPayload_23(__float_as_uint(p.x));
		optixSetPayload_24(__float_as_uint(p.y));
	}
	if constexpr (I == 24)
	{
		optixSetPayload_24(__float_as_uint(p.x));
		optixSetPayload_25(__float_as_uint(p.y));
	}
	if constexpr (I == 25)
	{
		optixSetPayload_25(__float_as_uint(p.x));
		optixSetPayload_26(__float_as_uint(p.y));
	}
	if constexpr (I == 26)
	{
		optixSetPayload_26(__float_as_uint(p.x));
		optixSetPayload_27(__float_as_uint(p.y));
	}
	if constexpr (I == 27)
	{
		optixSetPayload_27(__float_as_uint(p.x));
		optixSetPayload_28(__float_as_uint(p.y));
	}
	if constexpr (I == 28)
	{
		optixSetPayload_28(__float_as_uint(p.x));
		optixSetPayload_29(__float_as_uint(p.y));
	}
	if constexpr (I == 29)
	{
		optixSetPayload_29(__float_as_uint(p.x));
		optixSetPayload_30(__float_as_uint(p.y));
	}
	if constexpr (I == 30)
	{
		optixSetPayload_30(__float_as_uint(p.x));
		optixSetPayload_31(__float_as_uint(p.y));
	}
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE auto get_payload_float2() -> float2
{
	if constexpr (I == 0)
	{
		return { __uint_as_float(optixGetPayload_0()),__uint_as_float(optixGetPayload_1()) };
	}
	if constexpr (I == 1)
	{
		return { __uint_as_float(optixGetPayload_1()),__uint_as_float(optixGetPayload_2()) };
	}
	if constexpr (I == 2)
	{
		return { __uint_as_float(optixGetPayload_2()),__uint_as_float(optixGetPayload_3()) };
	}
	if constexpr (I == 3)
	{
		return { __uint_as_float(optixGetPayload_3()),__uint_as_float(optixGetPayload_4()) };
	}
	if constexpr (I == 4)
	{
		return { __uint_as_float(optixGetPayload_4()),__uint_as_float(optixGetPayload_5()) };
	}
	if constexpr (I == 5)
	{
		return { __uint_as_float(optixGetPayload_5()),__uint_as_float(optixGetPayload_6()) };
	}
	if constexpr (I == 6)
	{
		return { __uint_as_float(optixGetPayload_6()),__uint_as_float(optixGetPayload_7()) };
	}
	if constexpr (I == 7)
	{
		return { __uint_as_float(optixGetPayload_7()),__uint_as_float(optixGetPayload_8()) };
	}
	if constexpr (I == 8)
	{
		return { __uint_as_float(optixGetPayload_8()),__uint_as_float(optixGetPayload_9()) };
	}
	if constexpr (I == 9)
	{
		return { __uint_as_float(optixGetPayload_9()),__uint_as_float(optixGetPayload_10()) };
	}
	if constexpr (I == 10)
	{
		return { __uint_as_float(optixGetPayload_10()),__uint_as_float(optixGetPayload_11()) };
	}
	if constexpr (I == 11)
	{
		return { __uint_as_float(optixGetPayload_11()),__uint_as_float(optixGetPayload_12()) };
	}
	if constexpr (I == 12)
	{
		return { __uint_as_float(optixGetPayload_12()),__uint_as_float(optixGetPayload_13()) };
	}
	if constexpr (I == 13)
	{
		return { __uint_as_float(optixGetPayload_13()),__uint_as_float(optixGetPayload_14()) };
	}
	if constexpr (I == 14)
	{
		return { __uint_as_float(optixGetPayload_14()),__uint_as_float(optixGetPayload_15()) };
	}
	if constexpr (I == 15)
	{
		return { __uint_as_float(optixGetPayload_15()),__uint_as_float(optixGetPayload_16()) };
	}
	if constexpr (I == 16)
	{
		return { __uint_as_float(optixGetPayload_16()),__uint_as_float(optixGetPayload_17()) };
	}
	if constexpr (I == 17)
	{
		return { __uint_as_float(optixGetPayload_17()),__uint_as_float(optixGetPayload_18()) };
	}
	if constexpr (I == 18)
	{
		return { __uint_as_float(optixGetPayload_18()),__uint_as_float(optixGetPayload_19()) };
	}
	if constexpr (I == 19)
	{
		return { __uint_as_float(optixGetPayload_19()),__uint_as_float(optixGetPayload_20()) };
	}
	if constexpr (I == 20)
	{
		return { __uint_as_float(optixGetPayload_20()),__uint_as_float(optixGetPayload_21()) };
	}
	if constexpr (I == 21)
	{
		return { __uint_as_float(optixGetPayload_21()),__uint_as_float(optixGetPayload_22()) };
	}
	if constexpr (I == 22)
	{
		return { __uint_as_float(optixGetPayload_22()),__uint_as_float(optixGetPayload_23()) };
	}
	if constexpr (I == 23)
	{
		return { __uint_as_float(optixGetPayload_23()),__uint_as_float(optixGetPayload_24()) };
	}
	if constexpr (I == 24)
	{
		return { __uint_as_float(optixGetPayload_24()),__uint_as_float(optixGetPayload_25()) };
	}
	if constexpr (I == 25)
	{
		return { __uint_as_float(optixGetPayload_25()),__uint_as_float(optixGetPayload_26()) };
	}
	if constexpr (I == 26)
	{
		return { __uint_as_float(optixGetPayload_26()),__uint_as_float(optixGetPayload_27()) };
	}
	if constexpr (I == 27)
	{
		return { __uint_as_float(optixGetPayload_27()),__uint_as_float(optixGetPayload_28()) };
	}
	if constexpr (I == 28)
	{
		return { __uint_as_float(optixGetPayload_28()),__uint_as_float(optixGetPayload_29()) };
	}
	if constexpr (I == 29)
	{
		return { __uint_as_float(optixGetPayload_29()),__uint_as_float(optixGetPayload_30()) };
	}
	if constexpr (I == 30)
	{
		return { __uint_as_float(optixGetPayload_30()),__uint_as_float(optixGetPayload_31()) };
	}
	return {};
}
template<unsigned int I>
OTK_INLINE OTK_DEVICE void set_payload_float3(float3 p)
{
	if constexpr (I == 0)
	{
		optixSetPayload_0(__float_as_uint(p.x));
		optixSetPayload_1(__float_as_uint(p.y));
		optixSetPayload_2(__float_as_uint(p.z));
	}
	if constexpr (I == 1)
	{
		optixSetPayload_1(__float_as_uint(p.x));
		optixSetPayload_2(__float_as_uint(p.y));
		optixSetPayload_3(__float_as_uint(p.z));
	}
	if constexpr (I == 2)
	{
		optixSetPayload_2(__float_as_uint(p.x));
		optixSetPayload_3(__float_as_uint(p.y));
		optixSetPayload_4(__float_as_uint(p.z));
	}
	if constexpr (I == 3)
	{
		optixSetPayload_3(__float_as_uint(p.x));
		optixSetPayload_4(__float_as_uint(p.y));
		optixSetPayload_5(__float_as_uint(p.z));
	}
	if constexpr (I == 4)
	{
		optixSetPayload_4(__float_as_uint(p.x));
		optixSetPayload_5(__float_as_uint(p.y));
		optixSetPayload_6(__float_as_uint(p.z));
	}
	if constexpr (I == 5)
	{
		optixSetPayload_5(__float_as_uint(p.x));
		optixSetPayload_6(__float_as_uint(p.y));
		optixSetPayload_7(__float_as_uint(p.z));
	}
	if constexpr (I == 6)
	{
		optixSetPayload_6(__float_as_uint(p.x));
		optixSetPayload_7(__float_as_uint(p.y));
		optixSetPayload_8(__float_as_uint(p.z));
	}
	if constexpr (I == 7)
	{
		optixSetPayload_7(__float_as_uint(p.x));
		optixSetPayload_8(__float_as_uint(p.y));
		optixSetPayload_9(__float_as_uint(p.z));
	}
	if constexpr (I == 8)
	{
		optixSetPayload_8(__float_as_uint(p.x));
		optixSetPayload_9(__float_as_uint(p.y));
		optixSetPayload_10(__float_as_uint(p.z));
	}
	if constexpr (I == 9)
	{
		optixSetPayload_9(__float_as_uint(p.x));
		optixSetPayload_10(__float_as_uint(p.y));
		optixSetPayload_11(__float_as_uint(p.z));
	}
	if constexpr (I == 10)
	{
		optixSetPayload_10(__float_as_uint(p.x));
		optixSetPayload_11(__float_as_uint(p.y));
		optixSetPayload_12(__float_as_uint(p.z));
	}
	if constexpr (I == 11)
	{
		optixSetPayload_11(__float_as_uint(p.x));
		optixSetPayload_12(__float_as_uint(p.y));
		optixSetPayload_13(__float_as_uint(p.z));
	}
	if constexpr (I == 12)
	{
		optixSetPayload_12(__float_as_uint(p.x));
		optixSetPayload_13(__float_as_uint(p.y));
		optixSetPayload_14(__float_as_uint(p.z));
	}
	if constexpr (I == 13)
	{
		optixSetPayload_13(__float_as_uint(p.x));
		optixSetPayload_14(__float_as_uint(p.y));
		optixSetPayload_15(__float_as_uint(p.z));
	}
	if constexpr (I == 14)
	{
		optixSetPayload_14(__float_as_uint(p.x));
		optixSetPayload_15(__float_as_uint(p.y));
		optixSetPayload_16(__float_as_uint(p.z));
	}
	if constexpr (I == 15)
	{
		optixSetPayload_15(__float_as_uint(p.x));
		optixSetPayload_16(__float_as_uint(p.y));
		optixSetPayload_17(__float_as_uint(p.z));
	}
	if constexpr (I == 16)
	{
		optixSetPayload_16(__float_as_uint(p.x));
		optixSetPayload_17(__float_as_uint(p.y));
		optixSetPayload_18(__float_as_uint(p.z));
	}
	if constexpr (I == 17)
	{
		optixSetPayload_17(__float_as_uint(p.x));
		optixSetPayload_18(__float_as_uint(p.y));
		optixSetPayload_19(__float_as_uint(p.z));
	}
	if constexpr (I == 18)
	{
		optixSetPayload_18(__float_as_uint(p.x));
		optixSetPayload_19(__float_as_uint(p.y));
		optixSetPayload_20(__float_as_uint(p.z));
	}
	if constexpr (I == 19)
	{
		optixSetPayload_19(__float_as_uint(p.x));
		optixSetPayload_20(__float_as_uint(p.y));
		optixSetPayload_21(__float_as_uint(p.z));
	}
	if constexpr (I == 20)
	{
		optixSetPayload_20(__float_as_uint(p.x));
		optixSetPayload_21(__float_as_uint(p.y));
		optixSetPayload_22(__float_as_uint(p.z));
	}
	if constexpr (I == 21)
	{
		optixSetPayload_21(__float_as_uint(p.x));
		optixSetPayload_22(__float_as_uint(p.y));
		optixSetPayload_23(__float_as_uint(p.z));
	}
	if constexpr (I == 22)
	{
		optixSetPayload_22(__float_as_uint(p.x));
		optixSetPayload_23(__float_as_uint(p.y));
		optixSetPayload_24(__float_as_uint(p.z));
	}
	if constexpr (I == 23)
	{
		optixSetPayload_23(__float_as_uint(p.x));
		optixSetPayload_24(__float_as_uint(p.y));
		optixSetPayload_25(__float_as_uint(p.z));
	}
	if constexpr (I == 24)
	{
		optixSetPayload_24(__float_as_uint(p.x));
		optixSetPayload_25(__float_as_uint(p.y));
		optixSetPayload_26(__float_as_uint(p.z));
	}
	if constexpr (I == 25)
	{
		optixSetPayload_25(__float_as_uint(p.x));
		optixSetPayload_26(__float_as_uint(p.y));
		optixSetPayload_27(__float_as_uint(p.z));
	}
	if constexpr (I == 26)
	{
		optixSetPayload_26(__float_as_uint(p.x));
		optixSetPayload_27(__float_as_uint(p.y));
		optixSetPayload_28(__float_as_uint(p.z));
	}
	if constexpr (I == 27)
	{
		optixSetPayload_27(__float_as_uint(p.x));
		optixSetPayload_28(__float_as_uint(p.y));
		optixSetPayload_29(__float_as_uint(p.z));
	}
	if constexpr (I == 28)
	{
		optixSetPayload_28(__float_as_uint(p.x));
		optixSetPayload_29(__float_as_uint(p.y));
		optixSetPayload_30(__float_as_uint(p.z));
	}
	if constexpr (I == 29)
	{
		optixSetPayload_29(__float_as_uint(p.x));
		optixSetPayload_30(__float_as_uint(p.y));
		optixSetPayload_31(__float_as_uint(p.z));
	}
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE auto get_payload_float3() -> float3
{
	if constexpr (I == 0)
	{
		return { __uint_as_float(optixGetPayload_0()),__uint_as_float(optixGetPayload_1()),__uint_as_float(optixGetPayload_2()) };
	}
	if constexpr (I == 1)
	{
		return { __uint_as_float(optixGetPayload_1()),__uint_as_float(optixGetPayload_2()),__uint_as_float(optixGetPayload_3()) };
	}
	if constexpr (I == 2)
	{
		return { __uint_as_float(optixGetPayload_2()),__uint_as_float(optixGetPayload_3()),__uint_as_float(optixGetPayload_4()) };
	}
	if constexpr (I == 3)
	{
		return { __uint_as_float(optixGetPayload_3()),__uint_as_float(optixGetPayload_4()),__uint_as_float(optixGetPayload_5()) };
	}
	if constexpr (I == 4)
	{
		return { __uint_as_float(optixGetPayload_4()),__uint_as_float(optixGetPayload_5()),__uint_as_float(optixGetPayload_6()) };
	}
	if constexpr (I == 5)
	{
		return { __uint_as_float(optixGetPayload_5()),__uint_as_float(optixGetPayload_6()),__uint_as_float(optixGetPayload_7()) };
	}
	if constexpr (I == 6)
	{
		return { __uint_as_float(optixGetPayload_6()),__uint_as_float(optixGetPayload_7()),__uint_as_float(optixGetPayload_8()) };
	}
	if constexpr (I == 7)
	{
		return { __uint_as_float(optixGetPayload_7()),__uint_as_float(optixGetPayload_8()),__uint_as_float(optixGetPayload_9()) };
	}
	if constexpr (I == 8)
	{
		return { __uint_as_float(optixGetPayload_8()),__uint_as_float(optixGetPayload_9()),__uint_as_float(optixGetPayload_10()) };
	}
	if constexpr (I == 9)
	{
		return { __uint_as_float(optixGetPayload_9()),__uint_as_float(optixGetPayload_10()),__uint_as_float(optixGetPayload_11()) };
	}
	if constexpr (I == 10)
	{
		return { __uint_as_float(optixGetPayload_10()),__uint_as_float(optixGetPayload_11()),__uint_as_float(optixGetPayload_12()) };
	}
	if constexpr (I == 11)
	{
		return { __uint_as_float(optixGetPayload_11()),__uint_as_float(optixGetPayload_12()),__uint_as_float(optixGetPayload_13()) };
	}
	if constexpr (I == 12)
	{
		return { __uint_as_float(optixGetPayload_12()),__uint_as_float(optixGetPayload_13()),__uint_as_float(optixGetPayload_14()) };
	}
	if constexpr (I == 13)
	{
		return { __uint_as_float(optixGetPayload_13()),__uint_as_float(optixGetPayload_14()),__uint_as_float(optixGetPayload_15()) };
	}
	if constexpr (I == 14)
	{
		return { __uint_as_float(optixGetPayload_14()),__uint_as_float(optixGetPayload_15()),__uint_as_float(optixGetPayload_16()) };
	}
	if constexpr (I == 15)
	{
		return { __uint_as_float(optixGetPayload_15()),__uint_as_float(optixGetPayload_16()),__uint_as_float(optixGetPayload_17()) };
	}
	if constexpr (I == 16)
	{
		return { __uint_as_float(optixGetPayload_16()),__uint_as_float(optixGetPayload_17()),__uint_as_float(optixGetPayload_18()) };
	}
	if constexpr (I == 17)
	{
		return { __uint_as_float(optixGetPayload_17()),__uint_as_float(optixGetPayload_18()),__uint_as_float(optixGetPayload_19()) };
	}
	if constexpr (I == 18)
	{
		return { __uint_as_float(optixGetPayload_18()),__uint_as_float(optixGetPayload_19()),__uint_as_float(optixGetPayload_20()) };
	}
	if constexpr (I == 19)
	{
		return { __uint_as_float(optixGetPayload_19()),__uint_as_float(optixGetPayload_20()),__uint_as_float(optixGetPayload_21()) };
	}
	if constexpr (I == 20)
	{
		return { __uint_as_float(optixGetPayload_20()),__uint_as_float(optixGetPayload_21()),__uint_as_float(optixGetPayload_22()) };
	}
	if constexpr (I == 21)
	{
		return { __uint_as_float(optixGetPayload_21()),__uint_as_float(optixGetPayload_22()),__uint_as_float(optixGetPayload_23()) };
	}
	if constexpr (I == 22)
	{
		return { __uint_as_float(optixGetPayload_22()),__uint_as_float(optixGetPayload_23()),__uint_as_float(optixGetPayload_24()) };
	}
	if constexpr (I == 23)
	{
		return { __uint_as_float(optixGetPayload_23()),__uint_as_float(optixGetPayload_24()),__uint_as_float(optixGetPayload_25()) };
	}
	if constexpr (I == 24)
	{
		return { __uint_as_float(optixGetPayload_24()),__uint_as_float(optixGetPayload_25()),__uint_as_float(optixGetPayload_26()) };
	}
	if constexpr (I == 25)
	{
		return { __uint_as_float(optixGetPayload_25()),__uint_as_float(optixGetPayload_26()),__uint_as_float(optixGetPayload_27()) };
	}
	if constexpr (I == 26)
	{
		return { __uint_as_float(optixGetPayload_26()),__uint_as_float(optixGetPayload_27()),__uint_as_float(optixGetPayload_28()) };
	}
	if constexpr (I == 27)
	{
		return { __uint_as_float(optixGetPayload_27()),__uint_as_float(optixGetPayload_28()),__uint_as_float(optixGetPayload_29()) };
	}
	if constexpr (I == 28)
	{
		return { __uint_as_float(optixGetPayload_28()),__uint_as_float(optixGetPayload_29()),__uint_as_float(optixGetPayload_30()) };
	}
	if constexpr (I == 29)
	{
		return { __uint_as_float(optixGetPayload_29()),__uint_as_float(optixGetPayload_30()),__uint_as_float(optixGetPayload_31()) };
	}
	return {};
}
template<unsigned int I>
OTK_INLINE OTK_DEVICE void set_payload_float4(float4 p)
{
	if constexpr (I == 0)
	{
		optixSetPayload_0(__float_as_uint(p.x));
		optixSetPayload_1(__float_as_uint(p.y));
		optixSetPayload_2(__float_as_uint(p.z));
		optixSetPayload_3(__float_as_uint(p.w));
	}
	if constexpr (I == 1)
	{
		optixSetPayload_1(__float_as_uint(p.x));
		optixSetPayload_2(__float_as_uint(p.y));
		optixSetPayload_3(__float_as_uint(p.z));
		optixSetPayload_4(__float_as_uint(p.w));
	}
	if constexpr (I == 2)
	{
		optixSetPayload_2(__float_as_uint(p.x));
		optixSetPayload_3(__float_as_uint(p.y));
		optixSetPayload_4(__float_as_uint(p.z));
		optixSetPayload_5(__float_as_uint(p.w));
	}
	if constexpr (I == 3)
	{
		optixSetPayload_3(__float_as_uint(p.x));
		optixSetPayload_4(__float_as_uint(p.y));
		optixSetPayload_5(__float_as_uint(p.z));
		optixSetPayload_6(__float_as_uint(p.w));
	}
	if constexpr (I == 4)
	{
		optixSetPayload_4(__float_as_uint(p.x));
		optixSetPayload_5(__float_as_uint(p.y));
		optixSetPayload_6(__float_as_uint(p.z));
		optixSetPayload_7(__float_as_uint(p.w));
	}
	if constexpr (I == 5)
	{
		optixSetPayload_5(__float_as_uint(p.x));
		optixSetPayload_6(__float_as_uint(p.y));
		optixSetPayload_7(__float_as_uint(p.z));
		optixSetPayload_8(__float_as_uint(p.w));
	}
	if constexpr (I == 6)
	{
		optixSetPayload_6(__float_as_uint(p.x));
		optixSetPayload_7(__float_as_uint(p.y));
		optixSetPayload_8(__float_as_uint(p.z));
		optixSetPayload_9(__float_as_uint(p.w));
	}
	if constexpr (I == 7)
	{
		optixSetPayload_7(__float_as_uint(p.x));
		optixSetPayload_8(__float_as_uint(p.y));
		optixSetPayload_9(__float_as_uint(p.z));
		optixSetPayload_10(__float_as_uint(p.w));
	}
	if constexpr (I == 8)
	{
		optixSetPayload_8(__float_as_uint(p.x));
		optixSetPayload_9(__float_as_uint(p.y));
		optixSetPayload_10(__float_as_uint(p.z));
		optixSetPayload_11(__float_as_uint(p.w));
	}
	if constexpr (I == 9)
	{
		optixSetPayload_9(__float_as_uint(p.x));
		optixSetPayload_10(__float_as_uint(p.y));
		optixSetPayload_11(__float_as_uint(p.z));
		optixSetPayload_12(__float_as_uint(p.w));
	}
	if constexpr (I == 10)
	{
		optixSetPayload_10(__float_as_uint(p.x));
		optixSetPayload_11(__float_as_uint(p.y));
		optixSetPayload_12(__float_as_uint(p.z));
		optixSetPayload_13(__float_as_uint(p.w));
	}
	if constexpr (I == 11)
	{
		optixSetPayload_11(__float_as_uint(p.x));
		optixSetPayload_12(__float_as_uint(p.y));
		optixSetPayload_13(__float_as_uint(p.z));
		optixSetPayload_14(__float_as_uint(p.w));
	}
	if constexpr (I == 12)
	{
		optixSetPayload_12(__float_as_uint(p.x));
		optixSetPayload_13(__float_as_uint(p.y));
		optixSetPayload_14(__float_as_uint(p.z));
		optixSetPayload_15(__float_as_uint(p.w));
	}
	if constexpr (I == 13)
	{
		optixSetPayload_13(__float_as_uint(p.x));
		optixSetPayload_14(__float_as_uint(p.y));
		optixSetPayload_15(__float_as_uint(p.z));
		optixSetPayload_16(__float_as_uint(p.w));
	}
	if constexpr (I == 14)
	{
		optixSetPayload_14(__float_as_uint(p.x));
		optixSetPayload_15(__float_as_uint(p.y));
		optixSetPayload_16(__float_as_uint(p.z));
		optixSetPayload_17(__float_as_uint(p.w));
	}
	if constexpr (I == 15)
	{
		optixSetPayload_15(__float_as_uint(p.x));
		optixSetPayload_16(__float_as_uint(p.y));
		optixSetPayload_17(__float_as_uint(p.z));
		optixSetPayload_18(__float_as_uint(p.w));
	}
	if constexpr (I == 16)
	{
		optixSetPayload_16(__float_as_uint(p.x));
		optixSetPayload_17(__float_as_uint(p.y));
		optixSetPayload_18(__float_as_uint(p.z));
		optixSetPayload_19(__float_as_uint(p.w));
	}
	if constexpr (I == 17)
	{
		optixSetPayload_17(__float_as_uint(p.x));
		optixSetPayload_18(__float_as_uint(p.y));
		optixSetPayload_19(__float_as_uint(p.z));
		optixSetPayload_20(__float_as_uint(p.w));
	}
	if constexpr (I == 18)
	{
		optixSetPayload_18(__float_as_uint(p.x));
		optixSetPayload_19(__float_as_uint(p.y));
		optixSetPayload_20(__float_as_uint(p.z));
		optixSetPayload_21(__float_as_uint(p.w));
	}
	if constexpr (I == 19)
	{
		optixSetPayload_19(__float_as_uint(p.x));
		optixSetPayload_20(__float_as_uint(p.y));
		optixSetPayload_21(__float_as_uint(p.z));
		optixSetPayload_22(__float_as_uint(p.w));
	}
	if constexpr (I == 20)
	{
		optixSetPayload_20(__float_as_uint(p.x));
		optixSetPayload_21(__float_as_uint(p.y));
		optixSetPayload_22(__float_as_uint(p.z));
		optixSetPayload_23(__float_as_uint(p.w));
	}
	if constexpr (I == 21)
	{
		optixSetPayload_21(__float_as_uint(p.x));
		optixSetPayload_22(__float_as_uint(p.y));
		optixSetPayload_23(__float_as_uint(p.z));
		optixSetPayload_24(__float_as_uint(p.w));
	}
	if constexpr (I == 22)
	{
		optixSetPayload_22(__float_as_uint(p.x));
		optixSetPayload_23(__float_as_uint(p.y));
		optixSetPayload_24(__float_as_uint(p.z));
		optixSetPayload_25(__float_as_uint(p.w));
	}
	if constexpr (I == 23)
	{
		optixSetPayload_23(__float_as_uint(p.x));
		optixSetPayload_24(__float_as_uint(p.y));
		optixSetPayload_25(__float_as_uint(p.z));
		optixSetPayload_26(__float_as_uint(p.w));
	}
	if constexpr (I == 24)
	{
		optixSetPayload_24(__float_as_uint(p.x));
		optixSetPayload_25(__float_as_uint(p.y));
		optixSetPayload_26(__float_as_uint(p.z));
		optixSetPayload_27(__float_as_uint(p.w));
	}
	if constexpr (I == 25)
	{
		optixSetPayload_25(__float_as_uint(p.x));
		optixSetPayload_26(__float_as_uint(p.y));
		optixSetPayload_27(__float_as_uint(p.z));
		optixSetPayload_28(__float_as_uint(p.w));
	}
	if constexpr (I == 26)
	{
		optixSetPayload_26(__float_as_uint(p.x));
		optixSetPayload_27(__float_as_uint(p.y));
		optixSetPayload_28(__float_as_uint(p.z));
		optixSetPayload_29(__float_as_uint(p.w));
	}
	if constexpr (I == 27)
	{
		optixSetPayload_27(__float_as_uint(p.x));
		optixSetPayload_28(__float_as_uint(p.y));
		optixSetPayload_29(__float_as_uint(p.z));
		optixSetPayload_30(__float_as_uint(p.w));
	}
	if constexpr (I == 28)
	{
		optixSetPayload_28(__float_as_uint(p.x));
		optixSetPayload_29(__float_as_uint(p.y));
		optixSetPayload_30(__float_as_uint(p.z));
		optixSetPayload_31(__float_as_uint(p.w));
	}
}

template<unsigned int I>
OTK_INLINE OTK_DEVICE auto get_payload_float4() -> float4
{
	if constexpr (I == 0)
	{
		return { __uint_as_float(optixGetPayload_0()),__uint_as_float(optixGetPayload_1()),__uint_as_float(optixGetPayload_2()),__uint_as_float(optixGetPayload_3()) };
	}
	if constexpr (I == 1)
	{
		return { __uint_as_float(optixGetPayload_1()),__uint_as_float(optixGetPayload_2()),__uint_as_float(optixGetPayload_3()),__uint_as_float(optixGetPayload_4()) };
	}
	if constexpr (I == 2)
	{
		return { __uint_as_float(optixGetPayload_2()),__uint_as_float(optixGetPayload_3()),__uint_as_float(optixGetPayload_4()),__uint_as_float(optixGetPayload_5()) };
	}
	if constexpr (I == 3)
	{
		return { __uint_as_float(optixGetPayload_3()),__uint_as_float(optixGetPayload_4()),__uint_as_float(optixGetPayload_5()),__uint_as_float(optixGetPayload_6()) };
	}
	if constexpr (I == 4)
	{
		return { __uint_as_float(optixGetPayload_4()),__uint_as_float(optixGetPayload_5()),__uint_as_float(optixGetPayload_6()),__uint_as_float(optixGetPayload_7()) };
	}
	if constexpr (I == 5)
	{
		return { __uint_as_float(optixGetPayload_5()),__uint_as_float(optixGetPayload_6()),__uint_as_float(optixGetPayload_7()),__uint_as_float(optixGetPayload_8()) };
	}
	if constexpr (I == 6)
	{
		return { __uint_as_float(optixGetPayload_6()),__uint_as_float(optixGetPayload_7()),__uint_as_float(optixGetPayload_8()),__uint_as_float(optixGetPayload_9()) };
	}
	if constexpr (I == 7)
	{
		return { __uint_as_float(optixGetPayload_7()),__uint_as_float(optixGetPayload_8()),__uint_as_float(optixGetPayload_9()),__uint_as_float(optixGetPayload_10()) };
	}
	if constexpr (I == 8)
	{
		return { __uint_as_float(optixGetPayload_8()),__uint_as_float(optixGetPayload_9()),__uint_as_float(optixGetPayload_10()),__uint_as_float(optixGetPayload_11()) };
	}
	if constexpr (I == 9)
	{
		return { __uint_as_float(optixGetPayload_9()),__uint_as_float(optixGetPayload_10()),__uint_as_float(optixGetPayload_11()),__uint_as_float(optixGetPayload_12()) };
	}
	if constexpr (I == 10)
	{
		return { __uint_as_float(optixGetPayload_10()),__uint_as_float(optixGetPayload_11()),__uint_as_float(optixGetPayload_12()),__uint_as_float(optixGetPayload_13()) };
	}
	if constexpr (I == 11)
	{
		return { __uint_as_float(optixGetPayload_11()),__uint_as_float(optixGetPayload_12()),__uint_as_float(optixGetPayload_13()),__uint_as_float(optixGetPayload_14()) };
	}
	if constexpr (I == 12)
	{
		return { __uint_as_float(optixGetPayload_12()),__uint_as_float(optixGetPayload_13()),__uint_as_float(optixGetPayload_14()),__uint_as_float(optixGetPayload_15()) };
	}
	if constexpr (I == 13)
	{
		return { __uint_as_float(optixGetPayload_13()),__uint_as_float(optixGetPayload_14()),__uint_as_float(optixGetPayload_15()),__uint_as_float(optixGetPayload_16()) };
	}
	if constexpr (I == 14)
	{
		return { __uint_as_float(optixGetPayload_14()),__uint_as_float(optixGetPayload_15()),__uint_as_float(optixGetPayload_16()),__uint_as_float(optixGetPayload_17()) };
	}
	if constexpr (I == 15)
	{
		return { __uint_as_float(optixGetPayload_15()),__uint_as_float(optixGetPayload_16()),__uint_as_float(optixGetPayload_17()),__uint_as_float(optixGetPayload_18()) };
	}
	if constexpr (I == 16)
	{
		return { __uint_as_float(optixGetPayload_16()),__uint_as_float(optixGetPayload_17()),__uint_as_float(optixGetPayload_18()),__uint_as_float(optixGetPayload_19()) };
	}
	if constexpr (I == 17)
	{
		return { __uint_as_float(optixGetPayload_17()),__uint_as_float(optixGetPayload_18()),__uint_as_float(optixGetPayload_19()),__uint_as_float(optixGetPayload_20()) };
	}
	if constexpr (I == 18)
	{
		return { __uint_as_float(optixGetPayload_18()),__uint_as_float(optixGetPayload_19()),__uint_as_float(optixGetPayload_20()),__uint_as_float(optixGetPayload_21()) };
	}
	if constexpr (I == 19)
	{
		return { __uint_as_float(optixGetPayload_19()),__uint_as_float(optixGetPayload_20()),__uint_as_float(optixGetPayload_21()),__uint_as_float(optixGetPayload_22()) };
	}
	if constexpr (I == 20)
	{
		return { __uint_as_float(optixGetPayload_20()),__uint_as_float(optixGetPayload_21()),__uint_as_float(optixGetPayload_22()),__uint_as_float(optixGetPayload_23()) };
	}
	if constexpr (I == 21)
	{
		return { __uint_as_float(optixGetPayload_21()),__uint_as_float(optixGetPayload_22()),__uint_as_float(optixGetPayload_23()),__uint_as_float(optixGetPayload_24()) };
	}
	if constexpr (I == 22)
	{
		return { __uint_as_float(optixGetPayload_22()),__uint_as_float(optixGetPayload_23()),__uint_as_float(optixGetPayload_24()),__uint_as_float(optixGetPayload_25()) };
	}
	if constexpr (I == 23)
	{
		return { __uint_as_float(optixGetPayload_23()),__uint_as_float(optixGetPayload_24()),__uint_as_float(optixGetPayload_25()),__uint_as_float(optixGetPayload_26()) };
	}
	if constexpr (I == 24)
	{
		return { __uint_as_float(optixGetPayload_24()),__uint_as_float(optixGetPayload_25()),__uint_as_float(optixGetPayload_26()),__uint_as_float(optixGetPayload_27()) };
	}
	if constexpr (I == 25)
	{
		return { __uint_as_float(optixGetPayload_25()),__uint_as_float(optixGetPayload_26()),__uint_as_float(optixGetPayload_27()),__uint_as_float(optixGetPayload_28()) };
	}
	if constexpr (I == 26)
	{
		return { __uint_as_float(optixGetPayload_26()),__uint_as_float(optixGetPayload_27()),__uint_as_float(optixGetPayload_28()),__uint_as_float(optixGetPayload_29()) };
	}
	if constexpr (I == 27)
	{
		return { __uint_as_float(optixGetPayload_27()),__uint_as_float(optixGetPayload_28()),__uint_as_float(optixGetPayload_29()),__uint_as_float(optixGetPayload_30()) };
	}
	if constexpr (I == 28)
	{
		return { __uint_as_float(optixGetPayload_28()),__uint_as_float(optixGetPayload_29()),__uint_as_float(optixGetPayload_30()),__uint_as_float(optixGetPayload_31()) };
	}
	return {};
}
