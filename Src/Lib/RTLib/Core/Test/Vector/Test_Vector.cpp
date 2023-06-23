#include <stdexcept>
#include "gtest/gtest.h"

#include <RTLib/Core/Vector.h>
using namespace RTLib;

TEST(test_vector2, vector2_constructor) {
    EXPECT_EQ(Vector2_F16(1.0f, 2.0f), Vector2_F16(Vector3_F16(1.0f, 2.0f)));
    EXPECT_EQ(Vector2_F16(1.0f, 2.0f), Vector2_F16(Vector4_F16(1.0f, 2.0f)));

    EXPECT_EQ(Vector2_F32(1.0f, 2.0f), Vector2_F32(Vector3_F32(1.0f, 2.0f)));
    EXPECT_EQ(Vector2_F32(1.0f, 2.0f), Vector2_F32(Vector4_F32(1.0f, 2.0f)));
}

TEST(test_vector2, vector2_dot) {
    EXPECT_EQ(4.0f, Vector2_F16(1.0f).dot(Vector2_F16(2.0f)));
    EXPECT_EQ(8.5f, Vector2_F16(2.0f, 3.0f).dot(Vector2_F16(1.1f, 2.1f)));

    EXPECT_EQ(4.0f, Vector2_F32(1.0f).dot(Vector2_F32(2.0f)));
    EXPECT_EQ(8.5f, Vector2_F32(2.0f, 3.0f).dot(Vector2_F32(1.1f, 2.1f)));
}

TEST(test_vector3, vector3_constructor) {
    EXPECT_EQ(Vector3_F16(1.0f, 2.0f, 3.0f), Vector3_F16(Vector2_F16(1.0f, 2.0f), 3.0f));
    EXPECT_EQ(Vector3_F16(1.0f, 2.0f, 3.0f), Vector3_F16(1.0f, Vector2_F16(2.0f, 3.0f)));
    EXPECT_EQ(Vector3_F16(1.0f, 2.0f, 3.0f), Vector3_F16(Vector4_F16(1.0f, 2.0f, 3.0f)));

    EXPECT_EQ(Vector3_F32(1.0f, 2.0f,3.0f), Vector3_F32(Vector2_F32(1.0f, 2.0f),3.0f));
    EXPECT_EQ(Vector3_F32(1.0f, 2.0f,3.0f), Vector3_F32(1.0f,Vector2_F32(2.0f, 3.0f)));
    EXPECT_EQ(Vector3_F32(1.0f, 2.0f,3.0f), Vector3_F32(Vector4_F32(1.0f, 2.0f,3.0f)));
}

TEST(test_vector3, vector3_dot) {
    EXPECT_EQ(9.0f  , Vector3_F16(1.0f).dot(Vector3_F16(2.0f,3.0f,4.0f)));
    EXPECT_NEAR(12.22f, Vector3_F16(2.0f, 3.0f,1.2f).dot(Vector3_F16(1.1f, 2.1f,3.1f)),1e-2f);

    EXPECT_EQ(9.0f  , Vector3_F32(1.0f).dot(Vector3_F32(2.0f, 3.0f, 4.0f)));
    EXPECT_EQ(12.22f, Vector3_F32(2.0f, 3.0f, 1.2f).dot(Vector3_F32(1.1f, 2.1f, 3.1f)));
}

TEST(test_vector3, vector3_cross) {
    EXPECT_EQ(Vector3_F16(1.0f,0.0f,0.0f), Vector3_F16(0.0f,1.0f,0.0f).cross(Vector3_F16(0.0f,0.0f,1.0f)));
    EXPECT_EQ(Vector3_F16(0.0f,1.0f,0.0f), Vector3_F16(0.0f,0.0f,1.0f).cross(Vector3_F16(1.0f,0.0f,0.0f)));
    EXPECT_EQ(Vector3_F16(0.0f,0.0f,1.0f), Vector3_F16(1.0f,0.0f,0.0f).cross(Vector3_F16(0.0f,1.0f,0.0f)));

    EXPECT_EQ(Vector3_F32(1.0f,0.0f,0.0f), Vector3_F32(0.0f,1.0f,0.0f).cross(Vector3_F32(0.0f,0.0f,1.0f)));
    EXPECT_EQ(Vector3_F32(0.0f,1.0f,0.0f), Vector3_F32(0.0f,0.0f,1.0f).cross(Vector3_F32(1.0f,0.0f,0.0f)));
    EXPECT_EQ(Vector3_F32(0.0f,0.0f,1.0f), Vector3_F32(1.0f,0.0f,0.0f).cross(Vector3_F32(0.0f,1.0f,0.0f)));
}

TEST(test_vector3, vector3_cast) {
    EXPECT_EQ(Vector3_F32(1.0f,2.0f,3.0f),Vector3_F32(Vector3_F16(1.0f,2.0f,3.0f)));
}

TEST(test_vector3, vector3_angle_deg) {
    EXPECT_EQ(Float16(90.0f), Vector3_F16(0.0f, 1.0f, 0.0f).angle_deg(Vector3_F16(0.0f, 0.0f, 1.0f)));
    EXPECT_EQ(Float16(90.0f), Vector3_F16(0.0f, 0.0f, 1.0f).angle_deg(Vector3_F16(1.0f, 0.0f, 0.0f)));
    EXPECT_EQ(Float16(90.0f), Vector3_F16(1.0f, 0.0f, 0.0f).angle_deg(Vector3_F16(0.0f, 1.0f, 0.0f)));
}

TEST(test_vector3, vector3_normalize) {
    EXPECT_EQ(Vector3_F16(1.0f, 0.0f, 0.0f), Vector3_F16(2.0f, 0.0f, 0.0f).normalize());
    EXPECT_EQ(Vector3_F16(0.0f, 1.0f, 0.0f), Vector3_F16(0.0f, 3.0f, 0.0f).normalize());
    EXPECT_EQ(Vector3_F16(0.0f, 0.0f, 1.0f), Vector3_F16(0.0f, 0.0f, 4.0f).normalize());
}
