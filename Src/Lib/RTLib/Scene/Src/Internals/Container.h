#ifndef RTLIB_SCENE_INTERNAL_CONTAINER__H
#define RTLIB_SCENE_INTERNAL_CONTAINER__H
#include <queue>
#include <stack>
#include <deque>
#include <vector>
namespace RTLib
{
	namespace Scene
	{
		namespace Internals {
			template<typename T>
			class VectorStack : protected std::stack<T, std::vector<T>> {
			public:
				typedef std::size_t size_type;
				using std::stack<T, std::vector<T> >::empty;
				using std::stack<T, std::vector<T> >::size;
				using std::stack<T, std::vector<T> >::top;
				using std::stack<T, std::vector<T> >::push;
				using std::stack<T, std::vector<T> >::pop;

				auto get_vector() const noexcept -> const std::vector<T>& { return this->c; }
				auto get_vector()       noexcept ->       std::vector<T>& { return this->c; }
			};


			template <class T>
			class VectorQueue : protected std::queue<T, std::deque<T>> {
			public:
				using std::queue<T, std::deque<T>>::empty;
				using std::queue<T, std::deque<T>>::size;
				using std::queue<T, std::deque<T>>::front;
				using std::queue<T, std::deque<T>>::back;
				using std::queue<T, std::deque<T>>::push;
				using std::queue<T, std::deque<T>>::pop;

				auto get_deque() const noexcept -> const std::deque<T>& { return this->c; }
				auto get_deque()       noexcept ->       std::deque<T>& { return this->c; }
			};
		}
	}
}
#endif
