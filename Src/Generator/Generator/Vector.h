#ifndef RTLIB_CORE_VECTOR__H
#define RTLIB_CORE_VECTOR__H
#include <RTLib/Core/DataTypes.h>
namespace RTLib
{
	namespace Core
	{
		template <typename T> struct Vector2;
		template <typename T> struct Vector3;
		template <typename T> struct Vector4;
		
		template <typename T> struct Vector2
		{
			constexpr Vector2() noexcept;
			constexpr Vector2(const Vector2&) noexcept = default;

			constexpr Vector2& operator=(const Vector2&) noexcept = default;

			constexpr Vector2(T x_,T y_) noexcept:x{x_},y{y_}{}
			constexpr Vector2(T s) noexcept:x{s},y{s}{}

			template <typename U    > explicit constexpr Vector2(const Vector2<U>& v) noexcept:x{static_cast<T>(v.x)},y{static_cast<T>(v.y)}{}

			constexpr Bool operator==(const Vector2& v) const noexcept;
			constexpr Bool operator!=(const Vector2& v) const noexcept { return !(*this==v); }

			constexpr Vector2& operator+=(const Vector2&) noexcept;
			constexpr Vector2& operator-=(const Vector2&) noexcept;
			constexpr Vector2& operator*=(const Vector2&) noexcept;
			constexpr Vector2& operator/=(const Vector2&) noexcept;
			
			constexpr Vector2& operator*=(T s) noexcept;
			constexpr Vector2& operator/=(T s) noexcept;

			constexpr T dot(const Vector2& v) const noexcept{ return x*v.x+y*v.y; }
			constexpr T length_sqr() const noexcept { return dot(v); }
			T length() const noexcept { return RTLib::Core::sqrt(length_sqr()); }

			T x;
			T y;
		};
		template <typename T> struct Vector3
		{
			constexpr Vector3() noexcept;
			constexpr Vector3(const Vector3&) noexcept = default;

			constexpr Vector3& operator=(const Vector3&) noexcept = default;

			constexpr Vector3(T x_,T y_,T z_) noexcept:x{x_},y{y_},z{z_}{}
			constexpr Vector3(T s) noexcept:x{s},y{s},z{s}{}

			template <typename U = T> explicit constexpr Vector3(const Vector2<U>& v) noexcept:x{static_cast<T>(v.x)},y{static_cast<T>(v.y)},z{static_cast<T>(0)}{}
			template <typename U    > explicit constexpr Vector3(const Vector3<U>& v) noexcept:x{static_cast<T>(v.x)},y{static_cast<T>(v.y)},z{static_cast<T>(v.z)}{}

			constexpr Bool operator==(const Vector3& v) const noexcept;
			constexpr Bool operator!=(const Vector3& v) const noexcept { return !(*this==v); }

			constexpr Vector3& operator+=(const Vector3&) noexcept;
			constexpr Vector3& operator-=(const Vector3&) noexcept;
			constexpr Vector3& operator*=(const Vector3&) noexcept;
			constexpr Vector3& operator/=(const Vector3&) noexcept;
			
			constexpr Vector3& operator*=(T s) noexcept;
			constexpr Vector3& operator/=(T s) noexcept;

			constexpr T dot(const Vector3& v) const noexcept{ return x*v.x+y*v.y+z*v.z; }
			constexpr T length_sqr() const noexcept { return dot(v); }
			T length() const noexcept { return RTLib::Core::sqrt(length_sqr()); }

			T x;
			T y;
			T z;
		};
		template <typename T> struct Vector4
		{
			constexpr Vector4() noexcept;
			constexpr Vector4(const Vector4&) noexcept = default;

			constexpr Vector4& operator=(const Vector4&) noexcept = default;

			constexpr Vector4(T x_,T y_,T z_,T w_) noexcept:x{x_},y{y_},z{z_},w{w_}{}
			constexpr Vector4(T s) noexcept:x{s},y{s},z{s},w{s}{}

			template <typename U = T> explicit constexpr Vector4(const Vector2<U>& v) noexcept:x{static_cast<T>(v.x)},y{static_cast<T>(v.y)},z{static_cast<T>(0)},w{static_cast<T>(v.w)}{}
			template <typename U = T> explicit constexpr Vector4(const Vector3<U>& v) noexcept:x{static_cast<T>(v.x)},y{static_cast<T>(v.y)},z{static_cast<T>(v.z)},w{static_cast<T>(0)}{}
			template <typename U    > explicit constexpr Vector4(const Vector4<U>& v) noexcept:x{static_cast<T>(v.x)},y{static_cast<T>(v.y)},z{static_cast<T>(v.z)},w{static_cast<T>(v.w)}{}

			constexpr Bool operator==(const Vector4& v) const noexcept;
			constexpr Bool operator!=(const Vector4& v) const noexcept { return !(*this==v); }

			constexpr Vector4& operator+=(const Vector4&) noexcept;
			constexpr Vector4& operator-=(const Vector4&) noexcept;
			constexpr Vector4& operator*=(const Vector4&) noexcept;
			constexpr Vector4& operator/=(const Vector4&) noexcept;
			
			constexpr Vector4& operator*=(T s) noexcept;
			constexpr Vector4& operator/=(T s) noexcept;

			constexpr T dot(const Vector4& v) const noexcept{ return x*v.x+y*v.y+z*v.z+w*v.w; }
			constexpr T length_sqr() const noexcept { return dot(v); }
			T length() const noexcept { return RTLib::Core::sqrt(length_sqr()); }

			T x;
			T y;
			T z;
			T w;
		};

		template<typename T> constexpr Vector2<T>& Vector2<T>::operator+=(const Vector2<T>& v) noexcept
		{
			x+=v.x;
			y+=v.y;
			return *this;
		}
		template<typename T> constexpr Vector2<T>& Vector2<T>::operator-=(const Vector2<T>& v) noexcept
		{
			x-=v.x;
			y-=v.y;
			return *this;
		}
		template<typename T> constexpr Vector2<T>& Vector2<T>::operator*=(const Vector2<T>& v) noexcept
		{
			x*=v.x;
			y*=v.y;
			return *this;
		}
		template<typename T> constexpr Vector2<T>& Vector2<T>::operator/=(const Vector2<T>& v) noexcept
		{
			x/=v.x;
			y/=v.y;
			return *this;
		}
		
		template<typename T> constexpr Vector3<T>& Vector3<T>::operator+=(const Vector3<T>& v) noexcept
		{
			x+=v.x;
			y+=v.y;
			z+=v.z;
			return *this;
		}
		template<typename T> constexpr Vector3<T>& Vector3<T>::operator-=(const Vector3<T>& v) noexcept
		{
			x-=v.x;
			y-=v.y;
			z-=v.z;
			return *this;
		}
		template<typename T> constexpr Vector3<T>& Vector3<T>::operator*=(const Vector3<T>& v) noexcept
		{
			x*=v.x;
			y*=v.y;
			z*=v.z;
			return *this;
		}
		template<typename T> constexpr Vector3<T>& Vector3<T>::operator/=(const Vector3<T>& v) noexcept
		{
			x/=v.x;
			y/=v.y;
			z/=v.z;
			return *this;
		}
		
		template<typename T> constexpr Vector4<T>& Vector4<T>::operator+=(const Vector4<T>& v) noexcept
		{
			x+=v.x;
			y+=v.y;
			z+=v.z;
			w+=v.w;
			return *this;
		}
		template<typename T> constexpr Vector4<T>& Vector4<T>::operator-=(const Vector4<T>& v) noexcept
		{
			x-=v.x;
			y-=v.y;
			z-=v.z;
			w-=v.w;
			return *this;
		}
		template<typename T> constexpr Vector4<T>& Vector4<T>::operator*=(const Vector4<T>& v) noexcept
		{
			x*=v.x;
			y*=v.y;
			z*=v.z;
			w*=v.w;
			return *this;
		}
		template<typename T> constexpr Vector4<T>& Vector4<T>::operator/=(const Vector4<T>& v) noexcept
		{
			x/=v.x;
			y/=v.y;
			z/=v.z;
			w/=v.w;
			return *this;
		}
		
		template<typename T> constexpr Vector2<T>& Vector2<T>::operator*=(T s) noexcept
		{
			x*=s;
			y*=s;
			return *this;
		}
		template<typename T> constexpr Vector2<T>& Vector2<T>::operator/=(T s) noexcept
		{
			x/=s;
			y/=s;
			return *this;
		}

		template<typename T> constexpr Vector3<T>& Vector3<T>::operator*=(T s) noexcept
		{
			x*=s;
			y*=s;
			z*=s;
			return *this;
		}
		template<typename T> constexpr Vector3<T>& Vector3<T>::operator/=(T s) noexcept
		{
			x/=s;
			y/=s;
			z/=s;
			return *this;
		}

		template<typename T> constexpr Vector4<T>& Vector4<T>::operator*=(T s) noexcept
		{
			x*=s;
			y*=s;
			z*=s;
			w*=s;
			return *this;
		}
		template<typename T> constexpr Vector4<T>& Vector4<T>::operator/=(T s) noexcept
		{
			x/=s;
			y/=s;
			z/=s;
			w/=s;
			return *this;
		}

		template<typename T> constexpr Vector2<T> operator+(const Vector2<T>& v0, const Vector2<T>& v1) noexcept
		{
			return Vector2<T>(v0)+=v1;
		}
		template<typename T> constexpr Vector2<T> operator-(const Vector2<T>& v0, const Vector2<T>& v1) noexcept
		{
			return Vector2<T>(v0)-=v1;
		}
		template<typename T> constexpr Vector2<T> operator*(const Vector2<T>& v0, const Vector2<T>& v1) noexcept
		{
			return Vector2<T>(v0)*=v1;
		}
		template<typename T> constexpr Vector2<T> operator/(const Vector2<T>& v0, const Vector2<T>& v1) noexcept
		{
			return Vector2<T>(v0)/=v1;
		}
		
		template<typename T> constexpr Vector3<T> operator+(const Vector3<T>& v0, const Vector3<T>& v1) noexcept
		{
			return Vector3<T>(v0)+=v1;
		}
		template<typename T> constexpr Vector3<T> operator-(const Vector3<T>& v0, const Vector3<T>& v1) noexcept
		{
			return Vector3<T>(v0)-=v1;
		}
		template<typename T> constexpr Vector3<T> operator*(const Vector3<T>& v0, const Vector3<T>& v1) noexcept
		{
			return Vector3<T>(v0)*=v1;
		}
		template<typename T> constexpr Vector3<T> operator/(const Vector3<T>& v0, const Vector3<T>& v1) noexcept
		{
			return Vector3<T>(v0)/=v1;
		}
		
		template<typename T> constexpr Vector4<T> operator+(const Vector4<T>& v0, const Vector4<T>& v1) noexcept
		{
			return Vector4<T>(v0)+=v1;
		}
		template<typename T> constexpr Vector4<T> operator-(const Vector4<T>& v0, const Vector4<T>& v1) noexcept
		{
			return Vector4<T>(v0)-=v1;
		}
		template<typename T> constexpr Vector4<T> operator*(const Vector4<T>& v0, const Vector4<T>& v1) noexcept
		{
			return Vector4<T>(v0)*=v1;
		}
		template<typename T> constexpr Vector4<T> operator/(const Vector4<T>& v0, const Vector4<T>& v1) noexcept
		{
			return Vector4<T>(v0)/=v1;
		}
		
		template<typename T> constexpr Vector2<T> operator*=(const Vector2<T>& v0, T v1) noexcept
		{
			return Vector2<T>(v0)*=v1;
		}
		template<typename T> constexpr Vector2<T> operator*=(T v0, const Vector2<T>& v1) noexcept
		{
			return Vector2<T>(v1)*=v0;
		}
		template<typename T> constexpr Vector2<T> operator/=(const Vector2<T>& v0, T v1) noexcept
		{
			return Vector2<T>(v0)/=v1;
		}

		template<typename T> constexpr Vector3<T> operator*=(const Vector3<T>& v0, T v1) noexcept
		{
			return Vector3<T>(v0)*=v1;
		}
		template<typename T> constexpr Vector3<T> operator*=(T v0, const Vector3<T>& v1) noexcept
		{
			return Vector3<T>(v1)*=v0;
		}
		template<typename T> constexpr Vector3<T> operator/=(const Vector3<T>& v0, T v1) noexcept
		{
			return Vector3<T>(v0)/=v1;
		}

		template<typename T> constexpr Vector4<T> operator*=(const Vector4<T>& v0, T v1) noexcept
		{
			return Vector4<T>(v0)*=v1;
		}
		template<typename T> constexpr Vector4<T> operator*=(T v0, const Vector4<T>& v1) noexcept
		{
			return Vector4<T>(v1)*=v0;
		}
		template<typename T> constexpr Vector4<T> operator/=(const Vector4<T>& v0, T v1) noexcept
		{
			return Vector4<T>(v0)/=v1;
		}


		template<typename T> T dot(const Vector2& v0, const Vector2& v1) noexcept
		{
			return v0.dot(v1);
		}
		template<typename T> constexpr T length_sqr(const Vector2& v) noexcept
		{
			return v.length_sqr();
		}
		template<typename T> T length(const Vector2& v) noexcept
		{
			return v.length();
		}

		template<typename T> T dot(const Vector3& v0, const Vector3& v1) noexcept
		{
			return v0.dot(v1);
		}
		template<typename T> constexpr T length_sqr(const Vector3& v) noexcept
		{
			return v.length_sqr();
		}
		template<typename T> T length(const Vector3& v) noexcept
		{
			return v.length();
		}

		template<typename T> T dot(const Vector4& v0, const Vector4& v1) noexcept
		{
			return v0.dot(v1);
		}
		template<typename T> constexpr T length_sqr(const Vector4& v) noexcept
		{
			return v.length_sqr();
		}
		template<typename T> T length(const Vector4& v) noexcept
		{
			return v.length();
		}

	}
}

#endif