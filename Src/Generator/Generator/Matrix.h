#ifndef RTLIB_CORE_MATRIX__H
#define RTLIB_CORE_MATRIX__H
#include <RTLib/Core/Vector.h>
namespace RTLib
{
	namespace Core
	{
		template <typename T> struct Matrix2x2;
		template <typename T> struct Matrix3x3;
		template <typename T> struct Matrix4x4;

		template <typename T> struct Matrix2x2
		{
			constexpr Matrix2x2() noexcept;
			constexpr Matrix2x2(const Matrix2x2& ) = default;
			constexpr Matrix2x2& operator=(const Matrix2x2& ) = default;

			constexpr Matrix2x2(T s) noexcept;
			constexpr Matrix2x2(T m00,T m10,T m01,T m11) noexcept;
			constexpr Matrix2x2(const Vector2<T>& col0_,const Vector2<T>& col1_) noexcept;

			constexpr Matrix2x2 operator+() const noexcept;
			constexpr Matrix2x2 operator-() const noexcept;

			constexpr T operator()(Int32 r, Int32 c) const noexcept;

			constexpr Matrix2x2 operator+(const Matrix2x2&) const noexcept;
			constexpr Matrix2x2 operator-(const Matrix2x2&) const noexcept;
			constexpr Matrix2x2 operator*(const Matrix2x2&) const noexcept;
			constexpr Matrix2x2 operator*(T s) const noexcept;
			constexpr Matrix2x2 operator/(T s) const noexcept;
			constexpr Vector2<T> operator*(const Vector2<T>& v) const noexcept;

			constexpr Matrix2x2& operator+=(const Matrix2x2&) noexcept;
			constexpr Matrix2x2& operator-=(const Matrix2x2&) noexcept;
			constexpr Matrix2x2& operator*=(const Matrix2x2&) noexcept;
			constexpr Matrix2x2& operator/=(const Matrix2x2&) noexcept;
			constexpr Matrix2x2& operator*=(T s) noexcept;
			constexpr Matrix2x2& operator/=(T s) noexcept;

			constexpr Bool operator==(const Matrix2x2& m) const noexcept;
			constexpr Bool operator!=(const Matrix2x2& m) const noexcept;
			
			constexpr T determinant() const noexcept;
			
			constexpr Matrix2x2 inverse() const noexcept;
			constexpr Matrix2x2& inversed() noexcept;

			constexpr Matrix2x2 transpose() const noexcept;
			constexpr Matrix2x2& transposed() noexcept;

			constexpr Matrix2x2 inverse_transpose() const noexcept;
			constexpr Matrix2x2& inverse_transposed() noexcept;

			static constexpr Matrix2x2 identity() noexcept;
			static constexpr Matrix2x2 zeros() noexcept;
			
			RTLib::Core::Vector2<T> col0;
			RTLib::Core::Vector2<T> col1;
		};
		template <typename T> struct Matrix3x3
		{
			constexpr Matrix3x3() noexcept;
			constexpr Matrix3x3(const Matrix3x3& ) = default;
			constexpr Matrix3x3& operator=(const Matrix3x3& ) = default;

			constexpr Matrix3x3(T s) noexcept;
			constexpr Matrix3x3(T m00,T m10,T m20,T m01,T m11,T m21,T m02,T m12,T m22) noexcept;
			constexpr Matrix3x3(const Vector3<T>& col0_,const Vector3<T>& col1_,const Vector3<T>& col2_) noexcept;

			constexpr Matrix3x3 operator+() const noexcept;
			constexpr Matrix3x3 operator-() const noexcept;

			constexpr T operator()(Int32 r, Int32 c) const noexcept;

			constexpr Matrix3x3 operator+(const Matrix3x3&) const noexcept;
			constexpr Matrix3x3 operator-(const Matrix3x3&) const noexcept;
			constexpr Matrix3x3 operator*(const Matrix3x3&) const noexcept;
			constexpr Matrix3x3 operator*(T s) const noexcept;
			constexpr Matrix3x3 operator/(T s) const noexcept;
			constexpr Vector3<T> operator*(const Vector3<T>& v) const noexcept;

			constexpr Matrix3x3& operator+=(const Matrix3x3&) noexcept;
			constexpr Matrix3x3& operator-=(const Matrix3x3&) noexcept;
			constexpr Matrix3x3& operator*=(const Matrix3x3&) noexcept;
			constexpr Matrix3x3& operator/=(const Matrix3x3&) noexcept;
			constexpr Matrix3x3& operator*=(T s) noexcept;
			constexpr Matrix3x3& operator/=(T s) noexcept;

			constexpr Bool operator==(const Matrix3x3& m) const noexcept;
			constexpr Bool operator!=(const Matrix3x3& m) const noexcept;
			
			constexpr T determinant() const noexcept;
			
			constexpr Matrix3x3 inverse() const noexcept;
			constexpr Matrix3x3& inversed() noexcept;

			constexpr Matrix3x3 transpose() const noexcept;
			constexpr Matrix3x3& transposed() noexcept;

			constexpr Matrix3x3 inverse_transpose() const noexcept;
			constexpr Matrix3x3& inverse_transposed() noexcept;

			static constexpr Matrix3x3 identity() noexcept;
			static constexpr Matrix3x3 zeros() noexcept;
			static constexpr Matrix3x3 from_scale(const Vector2<T>& v) noexcept;
			static constexpr Matrix3x3 from_translate(const Vector2<T>& v) noexcept;
			
			RTLib::Core::Vector3<T> col0;
			RTLib::Core::Vector3<T> col1;
			RTLib::Core::Vector3<T> col2;
		};
		template <typename T> struct Matrix4x4
		{
			constexpr Matrix4x4() noexcept;
			constexpr Matrix4x4(const Matrix4x4& ) = default;
			constexpr Matrix4x4& operator=(const Matrix4x4& ) = default;

			constexpr Matrix4x4(T s) noexcept;
			constexpr Matrix4x4(T m00,T m10,T m20,T m30,T m01,T m11,T m21,T m31,T m02,T m12,T m22,T m32,T m03,T m13,T m23,T m33) noexcept;
			constexpr Matrix4x4(const Vector4<T>& col0_,const Vector4<T>& col1_,const Vector4<T>& col2_,const Vector4<T>& col3_) noexcept;

			constexpr Matrix4x4 operator+() const noexcept;
			constexpr Matrix4x4 operator-() const noexcept;

			constexpr T operator()(Int32 r, Int32 c) const noexcept;

			constexpr Matrix4x4 operator+(const Matrix4x4&) const noexcept;
			constexpr Matrix4x4 operator-(const Matrix4x4&) const noexcept;
			constexpr Matrix4x4 operator*(const Matrix4x4&) const noexcept;
			constexpr Matrix4x4 operator*(T s) const noexcept;
			constexpr Matrix4x4 operator/(T s) const noexcept;
			constexpr Vector4<T> operator*(const Vector4<T>& v) const noexcept;

			constexpr Matrix4x4& operator+=(const Matrix4x4&) noexcept;
			constexpr Matrix4x4& operator-=(const Matrix4x4&) noexcept;
			constexpr Matrix4x4& operator*=(const Matrix4x4&) noexcept;
			constexpr Matrix4x4& operator/=(const Matrix4x4&) noexcept;
			constexpr Matrix4x4& operator*=(T s) noexcept;
			constexpr Matrix4x4& operator/=(T s) noexcept;

			constexpr Bool operator==(const Matrix4x4& m) const noexcept;
			constexpr Bool operator!=(const Matrix4x4& m) const noexcept;
			
			constexpr T determinant() const noexcept;
			
			constexpr Matrix4x4 inverse() const noexcept;
			constexpr Matrix4x4& inversed() noexcept;

			constexpr Matrix4x4 transpose() const noexcept;
			constexpr Matrix4x4& transposed() noexcept;

			constexpr Matrix4x4 inverse_transpose() const noexcept;
			constexpr Matrix4x4& inverse_transposed() noexcept;

			static constexpr Matrix4x4 identity() noexcept;
			static constexpr Matrix4x4 zeros() noexcept;
			static constexpr Matrix4x4 from_scale(const Vector3<T>& v) noexcept;
			static constexpr Matrix4x4 from_translate(const Vector3<T>& v) noexcept;
			
			RTLib::Core::Vector4<T> col0;
			RTLib::Core::Vector4<T> col1;
			RTLib::Core::Vector4<T> col2;
			RTLib::Core::Vector4<T> col3;
		};

		template<typename T> constexpr Vector2<T> operator*(const Vector2<T>& v, const Matrix2x2<T>& m) noexcept;
		template<typename T> constexpr Matrix2x2<T> operator*(T s, const Matrix2x2<T>& m) noexcept;
		template<typename T> constexpr Vector3<T> operator*(const Vector3<T>& v, const Matrix3x3<T>& m) noexcept;
		template<typename T> constexpr Matrix3x3<T> operator*(T s, const Matrix3x3<T>& m) noexcept;
		template<typename T> constexpr Vector4<T> operator*(const Vector4<T>& v, const Matrix4x4<T>& m) noexcept;
		template<typename T> constexpr Matrix4x4<T> operator*(T s, const Matrix4x4<T>& m) noexcept;

		template<typename T> constexpr Matrix2x2<T>::Matrix2x2() noexcept: col0{},col1{}{}
		template<typename T> constexpr Matrix2x2<T>::Matrix2x2(T m00,T m10,T m01,T m11) noexcept: col0{m00,m10},col1{m01,m11}{}
		template<typename T> constexpr Matrix2x2<T>::Matrix2x2(const Vector2<T>& col0_,const Vector2<T>& col1_) noexcept: col0{col0_},col1{col1_}{}
		template<typename T> constexpr Matrix2x2<T>::Matrix2x2(T s) noexcept: col0{static_cast<T>(s),static_cast<T>(0)},col1{static_cast<T>(0),static_cast<T>(s)}{}
		template<typename T> constexpr Matrix3x3<T>::Matrix3x3() noexcept: col0{},col1{},col2{}{}
		template<typename T> constexpr Matrix3x3<T>::Matrix3x3(T m00,T m10,T m20,T m01,T m11,T m21,T m02,T m12,T m22) noexcept: col0{m00,m10,m20},col1{m01,m11,m21},col2{m02,m12,m22}{}
		template<typename T> constexpr Matrix3x3<T>::Matrix3x3(const Vector3<T>& col0_,const Vector3<T>& col1_,const Vector3<T>& col2_) noexcept: col0{col0_},col1{col1_},col2{col2_}{}
		template<typename T> constexpr Matrix3x3<T>::Matrix3x3(T s) noexcept: col0{static_cast<T>(s),static_cast<T>(0),static_cast<T>(0)},col1{static_cast<T>(0),static_cast<T>(s),static_cast<T>(0)},col2{static_cast<T>(0),static_cast<T>(0),static_cast<T>(s)}{}
		template<typename T> constexpr Matrix4x4<T>::Matrix4x4() noexcept: col0{},col1{},col2{},col3{}{}
		template<typename T> constexpr Matrix4x4<T>::Matrix4x4(T m00,T m10,T m20,T m30,T m01,T m11,T m21,T m31,T m02,T m12,T m22,T m32,T m03,T m13,T m23,T m33) noexcept: col0{m00,m10,m20,m30},col1{m01,m11,m21,m31},col2{m02,m12,m22,m32},col3{m03,m13,m23,m33}{}
		template<typename T> constexpr Matrix4x4<T>::Matrix4x4(const Vector4<T>& col0_,const Vector4<T>& col1_,const Vector4<T>& col2_,const Vector4<T>& col3_) noexcept: col0{col0_},col1{col1_},col2{col2_},col3{col3_}{}
		template<typename T> constexpr Matrix4x4<T>::Matrix4x4(T s) noexcept: col0{static_cast<T>(s),static_cast<T>(0),static_cast<T>(0),static_cast<T>(0)},col1{static_cast<T>(0),static_cast<T>(s),static_cast<T>(0),static_cast<T>(0)},col2{static_cast<T>(0),static_cast<T>(0),static_cast<T>(s),static_cast<T>(0)},col3{static_cast<T>(0),static_cast<T>(0),static_cast<T>(0),static_cast<T>(s)}{}

		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::operator+() const noexcept { return *this; }
		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::operator-() const noexcept { 
			return Matrix2x2<T>(
			-col0.x,
			-col0.y,
			-col1.x,
			-col1.y
			);
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::operator+() const noexcept { return *this; }
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::operator-() const noexcept { 
			return Matrix3x3<T>(
			-col0.x,
			-col0.y,
			-col0.z,
			-col1.x,
			-col1.y,
			-col1.z,
			-col2.x,
			-col2.y,
			-col2.z
			);
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::operator+() const noexcept { return *this; }
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::operator-() const noexcept { 
			return Matrix4x4<T>(
			-col0.x,
			-col0.y,
			-col0.z,
			-col0.w,
			-col1.x,
			-col1.y,
			-col1.z,
			-col1.w,
			-col2.x,
			-col2.y,
			-col2.z,
			-col2.w,
			-col3.x,
			-col3.y,
			-col3.z,
			-col3.w
			);
		}
	
		template<typename T> constexpr T Matrix2x2<T>::operator()(Int32 r, Int32 c) const noexcept {
			switch (c){
			case 0: 
				return get_by_index(col0,r);
			case 1: 
				return get_by_index(col1,r);
			default:
				return static_cast<T>(0);
			}
		}
		template<typename T> constexpr T Matrix3x3<T>::operator()(Int32 r, Int32 c) const noexcept {
			switch (c){
			case 0: 
				return get_by_index(col0,r);
			case 1: 
				return get_by_index(col1,r);
			case 2: 
				return get_by_index(col2,r);
			default:
				return static_cast<T>(0);
			}
		}
		template<typename T> constexpr T Matrix4x4<T>::operator()(Int32 r, Int32 c) const noexcept {
			switch (c){
			case 0: 
				return get_by_index(col0,r);
			case 1: 
				return get_by_index(col1,r);
			case 2: 
				return get_by_index(col2,r);
			case 3: 
				return get_by_index(col3,r);
			default:
				return static_cast<T>(0);
			}
		}
	
		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::operator+(const Matrix2x2<T>& v) const noexcept
		{
			return Matrix2x2<T>(
			col0.x + v.col0.x,
			col0.y + v.col0.y,
			col1.x + v.col1.x,
			col1.y + v.col1.y
			);
		}
		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::operator-(const Matrix2x2<T>& v) const noexcept
		{
			return Matrix2x2<T>(
			col0.x - v.col0.x,
			col0.y - v.col0.y,
			col1.x - v.col1.x,
			col1.y - v.col1.y
			);
		}
		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::operator*(const Matrix2x2<T>& v) const noexcept
		{
			return Matrix2x2<T>(
			col0.x*v.col0.x+col1.x*v.col0.y,
			col0.y*v.col0.x+col1.y*v.col0.y,
			col0.x*v.col1.x+col1.x*v.col1.y,
			col0.y*v.col1.x+col1.y*v.col1.y);
		}
		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::operator*(T s) const noexcept{
			return Matrix2x2<T>(col0.x*s,col0.y*s,col1.x*s,col1.y*s);
		}
		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::operator/(T s) const noexcept{
			return Matrix2x2<T>(col0.x/s,col0.y/s,col1.x/s,col1.y/s);
		}
		template<typename T> constexpr Vector2<T> Matrix2x2<T>::operator*(const Vector2<T>& v) const noexcept{
			Vector2<T> res = {};
			res.x = col0.x*v.x+col1.x*v.y;
			res.y = col0.y*v.x+col1.y*v.y;
			return res;
		}

		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::operator+(const Matrix3x3<T>& v) const noexcept
		{
			return Matrix3x3<T>(
			col0.x + v.col0.x,
			col0.y + v.col0.y,
			col0.z + v.col0.z,
			col1.x + v.col1.x,
			col1.y + v.col1.y,
			col1.z + v.col1.z,
			col2.x + v.col2.x,
			col2.y + v.col2.y,
			col2.z + v.col2.z
			);
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::operator-(const Matrix3x3<T>& v) const noexcept
		{
			return Matrix3x3<T>(
			col0.x - v.col0.x,
			col0.y - v.col0.y,
			col0.z - v.col0.z,
			col1.x - v.col1.x,
			col1.y - v.col1.y,
			col1.z - v.col1.z,
			col2.x - v.col2.x,
			col2.y - v.col2.y,
			col2.z - v.col2.z
			);
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::operator*(const Matrix3x3<T>& v) const noexcept
		{
			return Matrix3x3<T>(
			col0.x*v.col0.x+col1.x*v.col0.y+col2.x*v.col0.z,
			col0.y*v.col0.x+col1.y*v.col0.y+col2.y*v.col0.z,
			col0.z*v.col0.x+col1.z*v.col0.y+col2.z*v.col0.z,
			col0.x*v.col1.x+col1.x*v.col1.y+col2.x*v.col1.z,
			col0.y*v.col1.x+col1.y*v.col1.y+col2.y*v.col1.z,
			col0.z*v.col1.x+col1.z*v.col1.y+col2.z*v.col1.z,
			col0.x*v.col2.x+col1.x*v.col2.y+col2.x*v.col2.z,
			col0.y*v.col2.x+col1.y*v.col2.y+col2.y*v.col2.z,
			col0.z*v.col2.x+col1.z*v.col2.y+col2.z*v.col2.z);
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::operator*(T s) const noexcept{
			return Matrix3x3<T>(col0.x*s,col0.y*s,col0.z*s,col1.x*s,col1.y*s,col1.z*s,col2.x*s,col2.y*s,col2.z*s);
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::operator/(T s) const noexcept{
			return Matrix3x3<T>(col0.x/s,col0.y/s,col0.z/s,col1.x/s,col1.y/s,col1.z/s,col2.x/s,col2.y/s,col2.z/s);
		}
		template<typename T> constexpr Vector3<T> Matrix3x3<T>::operator*(const Vector3<T>& v) const noexcept{
			Vector3<T> res = {};
			res.x = col0.x*v.x+col1.x*v.y+col2.x*v.z;
			res.y = col0.y*v.x+col1.y*v.y+col2.y*v.z;
			res.z = col0.z*v.x+col1.z*v.y+col2.z*v.z;
			return res;
		}

		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::operator+(const Matrix4x4<T>& v) const noexcept
		{
			return Matrix4x4<T>(
			col0.x + v.col0.x,
			col0.y + v.col0.y,
			col0.z + v.col0.z,
			col0.w + v.col0.w,
			col1.x + v.col1.x,
			col1.y + v.col1.y,
			col1.z + v.col1.z,
			col1.w + v.col1.w,
			col2.x + v.col2.x,
			col2.y + v.col2.y,
			col2.z + v.col2.z,
			col2.w + v.col2.w,
			col3.x + v.col3.x,
			col3.y + v.col3.y,
			col3.z + v.col3.z,
			col3.w + v.col3.w
			);
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::operator-(const Matrix4x4<T>& v) const noexcept
		{
			return Matrix4x4<T>(
			col0.x - v.col0.x,
			col0.y - v.col0.y,
			col0.z - v.col0.z,
			col0.w - v.col0.w,
			col1.x - v.col1.x,
			col1.y - v.col1.y,
			col1.z - v.col1.z,
			col1.w - v.col1.w,
			col2.x - v.col2.x,
			col2.y - v.col2.y,
			col2.z - v.col2.z,
			col2.w - v.col2.w,
			col3.x - v.col3.x,
			col3.y - v.col3.y,
			col3.z - v.col3.z,
			col3.w - v.col3.w
			);
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::operator*(const Matrix4x4<T>& v) const noexcept
		{
			return Matrix4x4<T>(
			col0.x*v.col0.x+col1.x*v.col0.y+col2.x*v.col0.z+col3.x*v.col0.w,
			col0.y*v.col0.x+col1.y*v.col0.y+col2.y*v.col0.z+col3.y*v.col0.w,
			col0.z*v.col0.x+col1.z*v.col0.y+col2.z*v.col0.z+col3.z*v.col0.w,
			col0.w*v.col0.x+col1.w*v.col0.y+col2.w*v.col0.z+col3.w*v.col0.w,
			col0.x*v.col1.x+col1.x*v.col1.y+col2.x*v.col1.z+col3.x*v.col1.w,
			col0.y*v.col1.x+col1.y*v.col1.y+col2.y*v.col1.z+col3.y*v.col1.w,
			col0.z*v.col1.x+col1.z*v.col1.y+col2.z*v.col1.z+col3.z*v.col1.w,
			col0.w*v.col1.x+col1.w*v.col1.y+col2.w*v.col1.z+col3.w*v.col1.w,
			col0.x*v.col2.x+col1.x*v.col2.y+col2.x*v.col2.z+col3.x*v.col2.w,
			col0.y*v.col2.x+col1.y*v.col2.y+col2.y*v.col2.z+col3.y*v.col2.w,
			col0.z*v.col2.x+col1.z*v.col2.y+col2.z*v.col2.z+col3.z*v.col2.w,
			col0.w*v.col2.x+col1.w*v.col2.y+col2.w*v.col2.z+col3.w*v.col2.w,
			col0.x*v.col3.x+col1.x*v.col3.y+col2.x*v.col3.z+col3.x*v.col3.w,
			col0.y*v.col3.x+col1.y*v.col3.y+col2.y*v.col3.z+col3.y*v.col3.w,
			col0.z*v.col3.x+col1.z*v.col3.y+col2.z*v.col3.z+col3.z*v.col3.w,
			col0.w*v.col3.x+col1.w*v.col3.y+col2.w*v.col3.z+col3.w*v.col3.w);
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::operator*(T s) const noexcept{
			return Matrix4x4<T>(col0.x*s,col0.y*s,col0.z*s,col0.w*s,col1.x*s,col1.y*s,col1.z*s,col1.w*s,col2.x*s,col2.y*s,col2.z*s,col2.w*s,col3.x*s,col3.y*s,col3.z*s,col3.w*s);
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::operator/(T s) const noexcept{
			return Matrix4x4<T>(col0.x/s,col0.y/s,col0.z/s,col0.w/s,col1.x/s,col1.y/s,col1.z/s,col1.w/s,col2.x/s,col2.y/s,col2.z/s,col2.w/s,col3.x/s,col3.y/s,col3.z/s,col3.w/s);
		}
		template<typename T> constexpr Vector4<T> Matrix4x4<T>::operator*(const Vector4<T>& v) const noexcept{
			Vector4<T> res = {};
			res.x = col0.x*v.x+col1.x*v.y+col2.x*v.z+col3.x*v.w;
			res.y = col0.y*v.x+col1.y*v.y+col2.y*v.z+col3.y*v.w;
			res.z = col0.z*v.x+col1.z*v.y+col2.z*v.z+col3.z*v.w;
			res.w = col0.w*v.x+col1.w*v.y+col2.w*v.z+col3.w*v.w;
			return res;
		}


		template<typename T> constexpr Matrix2x2<T>& Matrix2x2<T>::operator+=(const Matrix2x2<T>& v) noexcept
		{
			col0.x+=v.col0.x;
			col0.y+=v.col0.y;
			col1.x+=v.col1.x;
			col1.y+=v.col1.y;
			return *This;
		}
		template<typename T> constexpr Matrix2x2<T>& Matrix2x2<T>::operator-=(const Matrix2x2<T>& v) noexcept
		{
			col0.x-=v.col0.x;
			col0.y-=v.col0.y;
			col1.x-=v.col1.x;
			col1.y-=v.col1.y;
			return *This;
		}
		template<typename T> constexpr Matrix2x2<T>& Matrix2x2<T>::operator*=(const Matrix2x2<T>& v) noexcept
		{
			T m00 = col0.x*v.col0.x+col1.x*v.col0.y;
			T m10 = col0.y*v.col0.x+col1.y*v.col0.y;
			T m01 = col0.x*v.col1.x+col1.x*v.col1.y;
			T m11 = col0.y*v.col1.x+col1.y*v.col1.y;
			col0.x = m00;
			col0.y = m10;
			col1.x = m01;
			col1.y = m11;
			col0.x*=v.col0.x;
			col0.y*=v.col0.y;
			col1.x*=v.col1.x;
			col1.y*=v.col1.y;
			return *This;
		}
		template<typename T> constexpr Matrix2x2<T>& Matrix2x2<T>::operator*=(T s) noexcept
		{
			col0.x*=s;
			col0.y*=s;
			col1.x*=s;
			col1.y*=s;
			return *This;
		}
		template<typename T> constexpr Matrix2x2<T>& Matrix2x2<T>::operator/=(T s) noexcept
		{
			col0.x/=s;
			col0.y/=s;
			col1.x/=s;
			col1.y/=s;
			return *This;
		}

		template<typename T> constexpr Matrix3x3<T>& Matrix3x3<T>::operator+=(const Matrix3x3<T>& v) noexcept
		{
			col0.x+=v.col0.x;
			col0.y+=v.col0.y;
			col0.z+=v.col0.z;
			col1.x+=v.col1.x;
			col1.y+=v.col1.y;
			col1.z+=v.col1.z;
			col2.x+=v.col2.x;
			col2.y+=v.col2.y;
			col2.z+=v.col2.z;
			return *This;
		}
		template<typename T> constexpr Matrix3x3<T>& Matrix3x3<T>::operator-=(const Matrix3x3<T>& v) noexcept
		{
			col0.x-=v.col0.x;
			col0.y-=v.col0.y;
			col0.z-=v.col0.z;
			col1.x-=v.col1.x;
			col1.y-=v.col1.y;
			col1.z-=v.col1.z;
			col2.x-=v.col2.x;
			col2.y-=v.col2.y;
			col2.z-=v.col2.z;
			return *This;
		}
		template<typename T> constexpr Matrix3x3<T>& Matrix3x3<T>::operator*=(const Matrix3x3<T>& v) noexcept
		{
			T m00 = col0.x*v.col0.x+col1.x*v.col0.y+col2.x*v.col0.z;
			T m10 = col0.y*v.col0.x+col1.y*v.col0.y+col2.y*v.col0.z;
			T m20 = col0.z*v.col0.x+col1.z*v.col0.y+col2.z*v.col0.z;
			T m01 = col0.x*v.col1.x+col1.x*v.col1.y+col2.x*v.col1.z;
			T m11 = col0.y*v.col1.x+col1.y*v.col1.y+col2.y*v.col1.z;
			T m21 = col0.z*v.col1.x+col1.z*v.col1.y+col2.z*v.col1.z;
			T m02 = col0.x*v.col2.x+col1.x*v.col2.y+col2.x*v.col2.z;
			T m12 = col0.y*v.col2.x+col1.y*v.col2.y+col2.y*v.col2.z;
			T m22 = col0.z*v.col2.x+col1.z*v.col2.y+col2.z*v.col2.z;
			col0.x = m00;
			col0.y = m10;
			col0.z = m20;
			col1.x = m01;
			col1.y = m11;
			col1.z = m21;
			col2.x = m02;
			col2.y = m12;
			col2.z = m22;
			col0.x*=v.col0.x;
			col0.y*=v.col0.y;
			col0.z*=v.col0.z;
			col1.x*=v.col1.x;
			col1.y*=v.col1.y;
			col1.z*=v.col1.z;
			col2.x*=v.col2.x;
			col2.y*=v.col2.y;
			col2.z*=v.col2.z;
			return *This;
		}
		template<typename T> constexpr Matrix3x3<T>& Matrix3x3<T>::operator*=(T s) noexcept
		{
			col0.x*=s;
			col0.y*=s;
			col0.z*=s;
			col1.x*=s;
			col1.y*=s;
			col1.z*=s;
			col2.x*=s;
			col2.y*=s;
			col2.z*=s;
			return *This;
		}
		template<typename T> constexpr Matrix3x3<T>& Matrix3x3<T>::operator/=(T s) noexcept
		{
			col0.x/=s;
			col0.y/=s;
			col0.z/=s;
			col1.x/=s;
			col1.y/=s;
			col1.z/=s;
			col2.x/=s;
			col2.y/=s;
			col2.z/=s;
			return *This;
		}

		template<typename T> constexpr Matrix4x4<T>& Matrix4x4<T>::operator+=(const Matrix4x4<T>& v) noexcept
		{
			col0.x+=v.col0.x;
			col0.y+=v.col0.y;
			col0.z+=v.col0.z;
			col0.w+=v.col0.w;
			col1.x+=v.col1.x;
			col1.y+=v.col1.y;
			col1.z+=v.col1.z;
			col1.w+=v.col1.w;
			col2.x+=v.col2.x;
			col2.y+=v.col2.y;
			col2.z+=v.col2.z;
			col2.w+=v.col2.w;
			col3.x+=v.col3.x;
			col3.y+=v.col3.y;
			col3.z+=v.col3.z;
			col3.w+=v.col3.w;
			return *This;
		}
		template<typename T> constexpr Matrix4x4<T>& Matrix4x4<T>::operator-=(const Matrix4x4<T>& v) noexcept
		{
			col0.x-=v.col0.x;
			col0.y-=v.col0.y;
			col0.z-=v.col0.z;
			col0.w-=v.col0.w;
			col1.x-=v.col1.x;
			col1.y-=v.col1.y;
			col1.z-=v.col1.z;
			col1.w-=v.col1.w;
			col2.x-=v.col2.x;
			col2.y-=v.col2.y;
			col2.z-=v.col2.z;
			col2.w-=v.col2.w;
			col3.x-=v.col3.x;
			col3.y-=v.col3.y;
			col3.z-=v.col3.z;
			col3.w-=v.col3.w;
			return *This;
		}
		template<typename T> constexpr Matrix4x4<T>& Matrix4x4<T>::operator*=(const Matrix4x4<T>& v) noexcept
		{
			T m00 = col0.x*v.col0.x+col1.x*v.col0.y+col2.x*v.col0.z+col3.x*v.col0.w;
			T m10 = col0.y*v.col0.x+col1.y*v.col0.y+col2.y*v.col0.z+col3.y*v.col0.w;
			T m20 = col0.z*v.col0.x+col1.z*v.col0.y+col2.z*v.col0.z+col3.z*v.col0.w;
			T m30 = col0.w*v.col0.x+col1.w*v.col0.y+col2.w*v.col0.z+col3.w*v.col0.w;
			T m01 = col0.x*v.col1.x+col1.x*v.col1.y+col2.x*v.col1.z+col3.x*v.col1.w;
			T m11 = col0.y*v.col1.x+col1.y*v.col1.y+col2.y*v.col1.z+col3.y*v.col1.w;
			T m21 = col0.z*v.col1.x+col1.z*v.col1.y+col2.z*v.col1.z+col3.z*v.col1.w;
			T m31 = col0.w*v.col1.x+col1.w*v.col1.y+col2.w*v.col1.z+col3.w*v.col1.w;
			T m02 = col0.x*v.col2.x+col1.x*v.col2.y+col2.x*v.col2.z+col3.x*v.col2.w;
			T m12 = col0.y*v.col2.x+col1.y*v.col2.y+col2.y*v.col2.z+col3.y*v.col2.w;
			T m22 = col0.z*v.col2.x+col1.z*v.col2.y+col2.z*v.col2.z+col3.z*v.col2.w;
			T m32 = col0.w*v.col2.x+col1.w*v.col2.y+col2.w*v.col2.z+col3.w*v.col2.w;
			T m03 = col0.x*v.col3.x+col1.x*v.col3.y+col2.x*v.col3.z+col3.x*v.col3.w;
			T m13 = col0.y*v.col3.x+col1.y*v.col3.y+col2.y*v.col3.z+col3.y*v.col3.w;
			T m23 = col0.z*v.col3.x+col1.z*v.col3.y+col2.z*v.col3.z+col3.z*v.col3.w;
			T m33 = col0.w*v.col3.x+col1.w*v.col3.y+col2.w*v.col3.z+col3.w*v.col3.w;
			col0.x = m00;
			col0.y = m10;
			col0.z = m20;
			col0.w = m30;
			col1.x = m01;
			col1.y = m11;
			col1.z = m21;
			col1.w = m31;
			col2.x = m02;
			col2.y = m12;
			col2.z = m22;
			col2.w = m32;
			col3.x = m03;
			col3.y = m13;
			col3.z = m23;
			col3.w = m33;
			col0.x*=v.col0.x;
			col0.y*=v.col0.y;
			col0.z*=v.col0.z;
			col0.w*=v.col0.w;
			col1.x*=v.col1.x;
			col1.y*=v.col1.y;
			col1.z*=v.col1.z;
			col1.w*=v.col1.w;
			col2.x*=v.col2.x;
			col2.y*=v.col2.y;
			col2.z*=v.col2.z;
			col2.w*=v.col2.w;
			col3.x*=v.col3.x;
			col3.y*=v.col3.y;
			col3.z*=v.col3.z;
			col3.w*=v.col3.w;
			return *This;
		}
		template<typename T> constexpr Matrix4x4<T>& Matrix4x4<T>::operator*=(T s) noexcept
		{
			col0.x*=s;
			col0.y*=s;
			col0.z*=s;
			col0.w*=s;
			col1.x*=s;
			col1.y*=s;
			col1.z*=s;
			col1.w*=s;
			col2.x*=s;
			col2.y*=s;
			col2.z*=s;
			col2.w*=s;
			col3.x*=s;
			col3.y*=s;
			col3.z*=s;
			col3.w*=s;
			return *This;
		}
		template<typename T> constexpr Matrix4x4<T>& Matrix4x4<T>::operator/=(T s) noexcept
		{
			col0.x/=s;
			col0.y/=s;
			col0.z/=s;
			col0.w/=s;
			col1.x/=s;
			col1.y/=s;
			col1.z/=s;
			col1.w/=s;
			col2.x/=s;
			col2.y/=s;
			col2.z/=s;
			col2.w/=s;
			col3.x/=s;
			col3.y/=s;
			col3.z/=s;
			col3.w/=s;
			return *This;
		}


		template<typename T> constexpr Bool Matrix2x2<T>::operator==(const Matrix2x2& m) const noexcept {
			return (col0==m.col0)&&(col1==m.col1);
		}
		template<typename T> constexpr Bool Matrix2x2<T>::operator!=(const Matrix2x2& m) const noexcept {
			return !(*this==m);
		}

		template<typename T> constexpr Bool Matrix3x3<T>::operator==(const Matrix3x3& m) const noexcept {
			return (col0==m.col0)&&(col1==m.col1)&&(col2==m.col2);
		}
		template<typename T> constexpr Bool Matrix3x3<T>::operator!=(const Matrix3x3& m) const noexcept {
			return !(*this==m);
		}

		template<typename T> constexpr Bool Matrix4x4<T>::operator==(const Matrix4x4& m) const noexcept {
			return (col0==m.col0)&&(col1==m.col1)&&(col2==m.col2)&&(col3==m.col3);
		}
		template<typename T> constexpr Bool Matrix4x4<T>::operator!=(const Matrix4x4& m) const noexcept {
			return !(*this==m);
		}


		template<typename T> constexpr Vector2<T> operator*(const Vector2<T>& v, const Matrix2x2<T>& m) noexcept{
			Vector2<T> res = {};
			res.x = m.col0.x*v.x+m.col0.y*v.y;
			res.y = m.col1.x*v.x+m.col1.y*v.y;
			return res;
		}
		template<typename T> constexpr Matrix2x2<T> operator*(T s, const Matrix2x2<T>& m) noexcept {
			return m * s;
		}
		template<typename T> constexpr Vector3<T> operator*(const Vector3<T>& v, const Matrix3x3<T>& m) noexcept{
			Vector3<T> res = {};
			res.x = m.col0.x*v.x+m.col0.y*v.y+m.col0.z*v.z;
			res.y = m.col1.x*v.x+m.col1.y*v.y+m.col1.z*v.z;
			res.z = m.col2.x*v.x+m.col2.y*v.y+m.col2.z*v.z;
			return res;
		}
		template<typename T> constexpr Matrix3x3<T> operator*(T s, const Matrix3x3<T>& m) noexcept {
			return m * s;
		}
		template<typename T> constexpr Vector4<T> operator*(const Vector4<T>& v, const Matrix4x4<T>& m) noexcept{
			Vector4<T> res = {};
			res.x = m.col0.x*v.x+m.col0.y*v.y+m.col0.z*v.z+m.col0.w*v.w;
			res.y = m.col1.x*v.x+m.col1.y*v.y+m.col1.z*v.z+m.col1.w*v.w;
			res.z = m.col2.x*v.x+m.col2.y*v.y+m.col2.z*v.z+m.col2.w*v.w;
			res.w = m.col3.x*v.x+m.col3.y*v.y+m.col3.z*v.z+m.col3.w*v.w;
			return res;
		}
		template<typename T> constexpr Matrix4x4<T> operator*(T s, const Matrix4x4<T>& m) noexcept {
			return m * s;
		}

		template<typename T> constexpr T Matrix2x2<T>::determinant() const noexcept{
			return col0.x * col1.y - col0.y * col1.x;
		
		}
		template<typename T> constexpr T Matrix3x3<T>::determinant() const noexcept{
		
			return col0.x * col1.y * col2.z+col1.x * col2.y * col0.z+col2.x * col0.y * col1.z-col0.x * col2.y * col1.z-col1.x * col0.y * col2.z-col2.x * col1.y * col0.z;
		}
		template<typename T> constexpr T Matrix4x4<T>::determinant() const noexcept{
		
			T A2323 = col2.z * col3.w - col3.z * col2.w ;
			T A1323 = col1.z * col3.w - col3.z * col1.w ;
			T A1223 = col1.z * col2.w - col2.z * col1.w ;
			T A0323 = col0.z * col3.w - col3.z * col0.w ;
			T A0223 = col0.z * col2.w - col2.z * col0.w ;
			T A0123 = col0.z * col1.w - col1.z * col0.w ;

			return col0.x * ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 ) 
			- col1.x * ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 ) 
			+ col2.x * ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 ) 
			- col3.x * ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 ) ;
		}

		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::inverse() const noexcept{
					T invDet = static_cast<T>(1)/(col0.x*col1.y-col0.y*col1.x);
			return Matrix2x2<T>(invDet*col1.y,-invDet*col0.y,-invDet*col1.x,invDet*col0.x);
								}
		template<typename T> constexpr Matrix2x2<T>& Matrix2x2<T>::inversed() noexcept{
					T invDet = static_cast<T>(1)/(col0.x*col1.y-col0.y*col1.x);
			T m00 = invDet*col1.y;
			T m10 =-invDet*col0.y;
			T m01 =-invDet*col1.x;
			T m11 = invDet*col0.x;
			col0.x = m00; col0.y = m10;
			col1.x = m01; col1.y = m11;
							return *this;
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::inverse() const noexcept{
								T invDet = static_cast<T>(1)/(col0.x * col1.y * col2.z+col1.x * col2.y * col0.z+col2.x * col0.y * col1.z-col0.x * col2.y * col1.z-col1.x * col0.y * col2.z-col2.x * col1.y * col0.z);
			return Matrix3x3<T>(
			+invDet * (col1.y * col2.z - col1.z * col2.y),
			-invDet * (col0.y * col2.z - col0.z * col2.y),
			+invDet * (col0.y * col1.z - col0.z * col1.y),
			-invDet * (col1.x * col2.z - col1.z * col2.x),
			+invDet * (col0.x * col2.z - col0.z * col2.x),
			-invDet * (col0.x * col1.z - col0.z * col1.x),
			+invDet * (col1.x * col2.y - col1.y * col2.x),
			-invDet * (col0.x * col2.y - col0.y * col2.x),
			+invDet * (col0.x * col1.y - col0.y * col1.x));
						}
		template<typename T> constexpr Matrix3x3<T>& Matrix3x3<T>::inversed() noexcept{
										T invDet = static_cast<T>(1)/(col0.x * col1.y * col2.z+col1.x * col2.y * col0.z+col2.x * col0.y * col1.z-col0.x * col2.y * col1.z-col1.x * col0.y * col2.z-col2.x * col1.y * col0.z);
			T m00 = +invDet * (col1.y * col2.z - col1.z * col2.y);
			T m01 = -invDet * (col0.y * col2.z - col0.z * col2.y);
			T m02 = +invDet * (col0.y * col1.z - col0.z * col1.y);
			T m10 = -invDet * (col1.x * col2.z - col1.z * col2.x);
			T m11 = +invDet * (col0.x * col2.z - col0.z * col2.x);
			T m12 = -invDet * (col0.x * col1.z - col0.z * col1.x);
			T m20 = +invDet * (col1.x * col2.y - col1.y * col2.x);
			T m21 = -invDet * (col0.x * col2.y - col0.y * col2.x);
			T m22 = +invDet * (col0.x * col1.y - col0.y * col1.x);
col0.x = m00; col1.x = m01; col2.x = m02;
col0.y = m10; col1.y = m11; col2.y = m12;
col0.z = m20; col1.z = m21; col2.z = m22;
					return *this;
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::inverse() const noexcept{
									
			T A2323 = col2.z * col3.w - col3.z * col2.w ;
			T A1323 = col1.z * col3.w - col3.z * col1.w ;
			T A1223 = col1.z * col2.w - col2.z * col1.w ;
			T A0323 = col0.z * col3.w - col3.z * col0.w ;
			T A0223 = col0.z * col2.w - col2.z * col0.w ;
			T A0123 = col0.z * col1.w - col1.z * col0.w ;
			T A2313 = col2.y * col3.w - col3.y * col2.w ;
			T A1313 = col1.y * col3.w - col3.y * col1.w ;
			T A1213 = col1.y * col2.w - col2.y * col1.w ;
			T A2312 = col2.y * col3.z - col3.y * col2.z ;
			T A1312 = col1.y * col3.z - col3.y * col1.z ;
			T A1212 = col1.y * col2.z - col2.y * col1.z ;
			T A0313 = col0.y * col3.w - col3.y * col0.w ;
			T A0213 = col0.y * col2.w - col2.y * col0.w ;
			T A0312 = col0.y * col3.z - col3.y * col0.z ;
			T A0212 = col0.y * col2.z - col2.y * col0.z ;
			T A0113 = col0.y * col1.w - col1.y * col0.w ;
			T A0112 = col0.y * col1.z - col1.y * col0.z ;

			T det = col0.x * ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 ) 
			- col1.x * ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 ) 
			+ col2.x * ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 ) 
			- col3.x * ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 ) ;
			det = 1 / det;

			return Matrix4x4<T>(
				det *   ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 ),
				det * - ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 ),
				det *   ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 ),
				det * - ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 ),
				
				det * - ( col1.x * A2323 - col2.x * A1323 + col3.x * A1223 ),
				det *   ( col0.x * A2323 - col2.x * A0323 + col3.x * A0223 ),
				det * - ( col0.x * A1323 - col1.x * A0323 + col3.x * A0123 ),
				det *   ( col0.x * A1223 - col1.x * A0223 + col2.x * A0123 ),
				
				det *   ( col1.x * A2313 - col2.x * A1313 + col3.x * A1213 ),
				det * - ( col0.x * A2313 - col2.x * A0313 + col3.x * A0213 ),
				det *   ( col0.x * A1313 - col1.x * A0313 + col3.x * A0113 ),
				det * - ( col0.x * A1213 - col1.x * A0213 + col2.x * A0113 ),

				det * - ( col1.x * A2312 - col2.x * A1312 + col3.x * A1212 ),
				det *   ( col0.x * A2312 - col2.x * A0312 + col3.x * A0212 ),
				det * - ( col0.x * A1312 - col1.x * A0312 + col3.x * A0112 ),
				det *   ( col0.x * A1212 - col1.x * A0212 + col2.x * A0112 )
			);
				}
		template<typename T> constexpr Matrix4x4<T>& Matrix4x4<T>::inversed() noexcept{
									
			T A2323 = col2.z * col3.w - col3.z * col2.w ;
			T A1323 = col1.z * col3.w - col3.z * col1.w ;
			T A1223 = col1.z * col2.w - col2.z * col1.w ;
			T A0323 = col0.z * col3.w - col3.z * col0.w ;
			T A0223 = col0.z * col2.w - col2.z * col0.w ;
			T A0123 = col0.z * col1.w - col1.z * col0.w ;
			T A2313 = col2.y * col3.w - col3.y * col2.w ;
			T A1313 = col1.y * col3.w - col3.y * col1.w ;
			T A1213 = col1.y * col2.w - col2.y * col1.w ;
			T A2312 = col2.y * col3.z - col3.y * col2.z ;
			T A1312 = col1.y * col3.z - col3.y * col1.z ;
			T A1212 = col1.y * col2.z - col2.y * col1.z ;
			T A0313 = col0.y * col3.w - col3.y * col0.w ;
			T A0213 = col0.y * col2.w - col2.y * col0.w ;
			T A0312 = col0.y * col3.z - col3.y * col0.z ;
			T A0212 = col0.y * col2.z - col2.y * col0.z ;
			T A0113 = col0.y * col1.w - col1.y * col0.w ;
			T A0112 = col0.y * col1.z - col1.y * col0.z ;

			T det = col0.x * ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 ) 
			- col1.x * ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 ) 
			+ col2.x * ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 ) 
			- col3.x * ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 ) ;
			det = 1 / det;

			col0.x = det *   ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 );
			col0.y = det * - ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 );
			col0.z = det *   ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 );
			col0.w = det * - ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 );
				
			col1.x = det * - ( col1.x * A2323 - col2.x * A1323 + col3.x * A1223 );
			col1.y = det *   ( col0.x * A2323 - col2.x * A0323 + col3.x * A0223 );
			col1.z = det * - ( col0.x * A1323 - col1.x * A0323 + col3.x * A0123 );
			col1.w = det *   ( col0.x * A1223 - col1.x * A0223 + col2.x * A0123 );
				
			col2.x = det *   ( col1.x * A2313 - col2.x * A1313 + col3.x * A1213 );
			col2.y = det * - ( col0.x * A2313 - col2.x * A0313 + col3.x * A0213 );
			col2.z = det *   ( col0.x * A1313 - col1.x * A0313 + col3.x * A0113 );
			col2.w = det * - ( col0.x * A1213 - col1.x * A0213 + col2.x * A0113 );

			col3.x = det * - ( col1.x * A2312 - col2.x * A1312 + col3.x * A1212 );
			col3.y = det *   ( col0.x * A2312 - col2.x * A0312 + col3.x * A0212 );
			col3.z = det * - ( col0.x * A1312 - col1.x * A0312 + col3.x * A0112 );
			col3.w = det *   ( col0.x * A1212 - col1.x * A0212 + col2.x * A0112 );
			;
			return *this;
		}
		
		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::transpose() const noexcept
		{
			return Matrix2x2<T>(col0.x,col1.x,col0.y,col1.y);
		}
		template<typename T> constexpr Matrix2x2<T>& Matrix2x2<T>::transposed() noexcept
		{
			T m00 = col0.x;T m10 = col1.x;
			T m01 = col0.y;T m11 = col1.y;
			col0.x = m00;col0.y = m10;
			col1.x = m01;col1.y = m11;
			return *this;
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::transpose() const noexcept
		{
			return Matrix3x3<T>(col0.x,col1.x,col2.x,col0.y,col1.y,col2.y,col0.z,col1.z,col2.z);
		}
		template<typename T> constexpr Matrix3x3<T>& Matrix3x3<T>::transposed() noexcept
		{
			T m00 = col0.x;T m10 = col1.x;T m20 = col2.x;
			T m01 = col0.y;T m11 = col1.y;T m21 = col2.y;
			T m02 = col0.z;T m12 = col1.z;T m22 = col2.z;
			col0.x = m00;col0.y = m10;col0.z = m20;
			col1.x = m01;col1.y = m11;col1.z = m21;
			col2.x = m02;col2.y = m12;col2.z = m22;
			return *this;
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::transpose() const noexcept
		{
			return Matrix4x4<T>(col0.x,col1.x,col2.x,col3.x,col0.y,col1.y,col2.y,col3.y,col0.z,col1.z,col2.z,col3.z,col0.w,col1.w,col2.w,col3.w);
		}
		template<typename T> constexpr Matrix4x4<T>& Matrix4x4<T>::transposed() noexcept
		{
			T m00 = col0.x;T m10 = col1.x;T m20 = col2.x;T m30 = col3.x;
			T m01 = col0.y;T m11 = col1.y;T m21 = col2.y;T m31 = col3.y;
			T m02 = col0.z;T m12 = col1.z;T m22 = col2.z;T m32 = col3.z;
			T m03 = col0.w;T m13 = col1.w;T m23 = col2.w;T m33 = col3.w;
			col0.x = m00;col0.y = m10;col0.z = m20;col0.w = m30;
			col1.x = m01;col1.y = m11;col1.z = m21;col1.w = m31;
			col2.x = m02;col2.y = m12;col2.z = m22;col2.w = m32;
			col3.x = m03;col3.y = m13;col3.z = m23;col3.w = m33;
			return *this;
		}

		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::inverse_transpose() const noexcept{
					T invDet = static_cast<T>(1)/(col0.x*col1.y-col0.y*col1.x);
			return Matrix2x2<T>(invDet*col1.y,-invDet*col1.x,-invDet*col0.y,invDet*col0.x);
								}
		template<typename T> constexpr Matrix2x2<T>& Matrix2x2<T>::inverse_transposed() noexcept{
					T invDet = static_cast<T>(1)/(col0.x*col1.y-col0.y*col1.x);
			T m00 = invDet*col1.y;
			T m01 =-invDet*col0.y;
			T m10 =-invDet*col1.x;
			T m11 = invDet*col0.x;
			col0.x = m00; col0.y = m10;
			col1.x = m01; col1.y = m11;
							return *this;
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::inverse_transpose() const noexcept{
								T invDet = static_cast<T>(1)/(col0.x * col1.y * col2.z+col1.x * col2.y * col0.z+col2.x * col0.y * col1.z-col0.x * col2.y * col1.z-col1.x * col0.y * col2.z-col2.x * col1.y * col0.z);
			return Matrix3x3<T>(
			+invDet * (col1.y * col2.z - col1.z * col2.y),
			-invDet * (col1.x * col2.z - col1.z * col2.x),
			+invDet * (col1.x * col2.y - col1.y * col2.x),
			-invDet * (col0.y * col2.z - col0.z * col2.y),
			+invDet * (col0.x * col2.z - col0.z * col2.x),
			-invDet * (col0.x * col2.y - col0.y * col2.x),
			+invDet * (col0.y * col1.z - col0.z * col1.y),
			-invDet * (col0.x * col1.z - col0.z * col1.x),
			+invDet * (col0.x * col1.y - col0.y * col1.x));
						}
		template<typename T> constexpr Matrix3x3<T>& Matrix3x3<T>::inverse_transposed() noexcept{
										T invDet = static_cast<T>(1)/(col0.x * col1.y * col2.z+col1.x * col2.y * col0.z+col2.x * col0.y * col1.z-col0.x * col2.y * col1.z-col1.x * col0.y * col2.z-col2.x * col1.y * col0.z);
			T m00 = +invDet * (col1.y * col2.z - col1.z * col2.y);
			T m10 = -invDet * (col1.x * col2.z - col1.z * col2.x);
			T m20 = +invDet * (col1.x * col2.y - col1.y * col2.x);
			T m01 = -invDet * (col0.y * col2.z - col0.z * col2.y);
			T m11 = +invDet * (col0.x * col2.z - col0.z * col2.x);
			T m21 = -invDet * (col0.x * col2.y - col0.y * col2.x);
			T m02 = +invDet * (col0.y * col1.z - col0.z * col1.y);
			T m12 = -invDet * (col0.x * col1.z - col0.z * col1.x);
			T m22 = +invDet * (col0.x * col1.y - col0.y * col1.x);
col0.x = m00; col1.x = m01; col2.x = m02;
col0.y = m10; col1.y = m11; col2.y = m12;
col0.z = m20; col1.z = m21; col2.z = m22;
					return *this;
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::inverse_transpose() const noexcept{
									
			T A2323 = col2.z * col3.w - col3.z * col2.w ;
			T A1323 = col1.z * col3.w - col3.z * col1.w ;
			T A1223 = col1.z * col2.w - col2.z * col1.w ;
			T A0323 = col0.z * col3.w - col3.z * col0.w ;
			T A0223 = col0.z * col2.w - col2.z * col0.w ;
			T A0123 = col0.z * col1.w - col1.z * col0.w ;
			T A2313 = col2.y * col3.w - col3.y * col2.w ;
			T A1313 = col1.y * col3.w - col3.y * col1.w ;
			T A1213 = col1.y * col2.w - col2.y * col1.w ;
			T A2312 = col2.y * col3.z - col3.y * col2.z ;
			T A1312 = col1.y * col3.z - col3.y * col1.z ;
			T A1212 = col1.y * col2.z - col2.y * col1.z ;
			T A0313 = col0.y * col3.w - col3.y * col0.w ;
			T A0213 = col0.y * col2.w - col2.y * col0.w ;
			T A0312 = col0.y * col3.z - col3.y * col0.z ;
			T A0212 = col0.y * col2.z - col2.y * col0.z ;
			T A0113 = col0.y * col1.w - col1.y * col0.w ;
			T A0112 = col0.y * col1.z - col1.y * col0.z ;

			T det = col0.x * ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 ) 
			- col1.x * ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 ) 
			+ col2.x * ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 ) 
			- col3.x * ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 ) ;
			det = 1 / det;

			return Matrix4x4<T>(
			det *   ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 ),
			det * - ( col1.x * A2323 - col2.x * A1323 + col3.x * A1223 ),
			det *   ( col1.x * A2313 - col2.x * A1313 + col3.x * A1213 ),
			det * - ( col1.x * A2312 - col2.x * A1312 + col3.x * A1212 ),

			det * - ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 ),
			det *   ( col0.x * A2323 - col2.x * A0323 + col3.x * A0223 ),
			det * - ( col0.x * A2313 - col2.x * A0313 + col3.x * A0213 ),
			det *   ( col0.x * A2312 - col2.x * A0312 + col3.x * A0212 ),

			det *   ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 ),
			det * - ( col0.x * A1323 - col1.x * A0323 + col3.x * A0123 ),
			det *   ( col0.x * A1313 - col1.x * A0313 + col3.x * A0113 ),
			det * - ( col0.x * A1312 - col1.x * A0312 + col3.x * A0112 ),

			det * - ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 ),
			det *   ( col0.x * A1223 - col1.x * A0223 + col2.x * A0123 ),
			det * - ( col0.x * A1213 - col1.x * A0213 + col2.x * A0113 ),
			det *   ( col0.x * A1212 - col1.x * A0212 + col2.x * A0112 )
			);
				}
		template<typename T> constexpr Matrix4x4<T>& Matrix4x4<T>::inverse_transposed() noexcept{
									
			T A2323 = col2.z * col3.w - col3.z * col2.w ;
			T A1323 = col1.z * col3.w - col3.z * col1.w ;
			T A1223 = col1.z * col2.w - col2.z * col1.w ;
			T A0323 = col0.z * col3.w - col3.z * col0.w ;
			T A0223 = col0.z * col2.w - col2.z * col0.w ;
			T A0123 = col0.z * col1.w - col1.z * col0.w ;
			T A2313 = col2.y * col3.w - col3.y * col2.w ;
			T A1313 = col1.y * col3.w - col3.y * col1.w ;
			T A1213 = col1.y * col2.w - col2.y * col1.w ;
			T A2312 = col2.y * col3.z - col3.y * col2.z ;
			T A1312 = col1.y * col3.z - col3.y * col1.z ;
			T A1212 = col1.y * col2.z - col2.y * col1.z ;
			T A0313 = col0.y * col3.w - col3.y * col0.w ;
			T A0213 = col0.y * col2.w - col2.y * col0.w ;
			T A0312 = col0.y * col3.z - col3.y * col0.z ;
			T A0212 = col0.y * col2.z - col2.y * col0.z ;
			T A0113 = col0.y * col1.w - col1.y * col0.w ;
			T A0112 = col0.y * col1.z - col1.y * col0.z ;

			T det = col0.x * ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 ) 
			- col1.x * ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 ) 
			+ col2.x * ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 ) 
			- col3.x * ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 ) ;
			det = 1 / det;

			col0.x = det *   ( col1.y * A2323 - col2.y * A1323 + col3.y * A1223 );
			col0.y = det * - ( col1.x * A2323 - col2.x * A1323 + col3.x * A1223 );
			col0.z = det *   ( col1.x * A2313 - col2.x * A1313 + col3.x * A1213 );
			col0.w = det * - ( col1.x * A2312 - col2.x * A1312 + col3.x * A1212 );

			col1.x = det * - ( col0.y * A2323 - col2.y * A0323 + col3.y * A0223 );
			col1.y = det *   ( col0.x * A2323 - col2.x * A0323 + col3.x * A0223 );
			col1.z = det * - ( col0.x * A2313 - col2.x * A0313 + col3.x * A0213 );
			col1.w = det *   ( col0.x * A2312 - col2.x * A0312 + col3.x * A0212 );

			col2.x = det *   ( col0.y * A1323 - col1.y * A0323 + col3.y * A0123 );
			col2.y = det * - ( col0.x * A1323 - col1.x * A0323 + col3.x * A0123 );
			col2.z = det *   ( col0.x * A1313 - col1.x * A0313 + col3.x * A0113 );
			col2.w = det * - ( col0.x * A1312 - col1.x * A0312 + col3.x * A0112 );

			col3.x = det * - ( col0.y * A1223 - col1.y * A0223 + col2.y * A0123 );
			col3.y = det *   ( col0.x * A1223 - col1.x * A0223 + col2.x * A0123 );
			col3.z = det * - ( col0.x * A1213 - col1.x * A0213 + col2.x * A0113 );
			col3.w = det *   ( col0.x * A1212 - col1.x * A0212 + col2.x * A0112 );
			;
			return *this;
		}

		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::identity() noexcept
		{
			return Matrix2x2(static_cast<T>(1));
		}
		template<typename T> constexpr Matrix2x2<T> Matrix2x2<T>::zeros() noexcept
		{
			return Matrix2x2();
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::identity() noexcept
		{
			return Matrix3x3(static_cast<T>(1));
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::zeros() noexcept
		{
			return Matrix3x3();
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::from_scale(const Vector2<T>& v) noexcept
		{
			return Matrix3x3<T>(
			v.x,
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			v.y,
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(1)
			);
		}
		template<typename T> constexpr Matrix3x3<T> Matrix3x3<T>::from_translate(const Vector2<T>& v) noexcept
		{
			return Matrix3x3<T>(
			static_cast<T>(1),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(1),
			static_cast<T>(0),
			v.x,
			v.y,
			static_cast<T>(1)
			);
		}

		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::identity() noexcept
		{
			return Matrix4x4(static_cast<T>(1));
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::zeros() noexcept
		{
			return Matrix4x4();
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::from_scale(const Vector3<T>& v) noexcept
		{
			return Matrix4x4<T>(
			v.x,
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			v.y,
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			v.z,
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(1)
			);
		}
		template<typename T> constexpr Matrix4x4<T> Matrix4x4<T>::from_translate(const Vector3<T>& v) noexcept
		{
			return Matrix4x4<T>(
			static_cast<T>(1),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(1),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(0),
			static_cast<T>(1),
			static_cast<T>(0),
			v.x,
			v.y,
			v.z,
			static_cast<T>(1)
			);
		}


		using Matrix2x2_I8  = Matrix2x2<Int8 >;
		using Matrix2x2_I16 = Matrix2x2<Int16>;
		using Matrix2x2_I32 = Matrix2x2<Int32>;
		using Matrix2x2_I64 = Matrix2x2<Int64>;

		using Matrix2x2_U8  = Matrix2x2<UInt8>;
		using Matrix2x2_U16 = Matrix2x2<UInt16>;
		using Matrix2x2_U32 = Matrix2x2<UInt32>;
		using Matrix2x2_U64 = Matrix2x2<UInt64>;

		using Matrix2x2_F16 = Matrix2x2<Float16>;
		using Matrix2x2_F32 = Matrix2x2<Float32>;
		using Matrix2x2_F64 = Matrix2x2<Float64>;

		using Matrix3x3_I8  = Matrix3x3<Int8 >;
		using Matrix3x3_I16 = Matrix3x3<Int16>;
		using Matrix3x3_I32 = Matrix3x3<Int32>;
		using Matrix3x3_I64 = Matrix3x3<Int64>;

		using Matrix3x3_U8 = Matrix3x3<UInt8>;
		using Matrix3x3_U16 = Matrix3x3<UInt16>;
		using Matrix3x3_U32 = Matrix3x3<UInt32>;
		using Matrix3x3_U64 = Matrix3x3<UInt64>;

		using Matrix3x3_F16 = Matrix3x3<Float16>;
		using Matrix3x3_F32 = Matrix3x3<Float32>;
		using Matrix3x3_F64 = Matrix3x3<Float64>;

		using Matrix4x4_I8  = Matrix4x4<Int8 >;
		using Matrix4x4_I16 = Matrix4x4<Int16>;
		using Matrix4x4_I32 = Matrix4x4<Int32>;
		using Matrix4x4_I64 = Matrix4x4<Int64>;

		using Matrix4x4_U8  = Matrix4x4<UInt8>;
		using Matrix4x4_U16 = Matrix4x4<UInt16>;
		using Matrix4x4_U32 = Matrix4x4<UInt32>;
		using Matrix4x4_U64 = Matrix4x4<UInt64>;

		using Matrix4x4_F16 = Matrix4x4<Float16>;
		using Matrix4x4_F32 = Matrix4x4<Float32>;
		using Matrix4x4_F64 = Matrix4x4<Float64>;
	}
}
#endif