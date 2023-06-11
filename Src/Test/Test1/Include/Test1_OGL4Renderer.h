#ifndef TEST_TEST1_OGL4_RENDERER__H
#define TEST_TEST1_OGL4_RENDERER__H
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <memory>
namespace Test1
{
	struct OGL4Renderer 
	{
		 OGL4Renderer(GLFWwindow* window);
		~OGL4Renderer();

		void init();

		void free();

		void render();

		bool resize(int fb_width, int fb_height);

		auto get_window() const noexcept -> GLFWwindow*;

		auto get_pbo() const noexcept -> GLuint;

		void set_current();

		auto get_fb_width() const noexcept -> int { return m_FbWidth; }
		auto get_fb_height() const noexcept -> int { return m_FbHeight; }
	private:

		void init_vbo();
		void free_vbo();

		void init_ibo();
		void free_ibo();

		void init_vao();
		void free_vao();

		void init_pbo();
		void free_pbo();

		void init_tex();
		void free_tex();

		void init_prg();
		void free_prg();

	private:
		GLFWwindow* m_Window = nullptr;
		std::unique_ptr<GladGLContext> m_GladContext = nullptr;

		int m_FbWidth  = 0;
		int m_FbHeight = 0;
		
		GLuint m_Vao = 0;
		GLuint m_Vbo = 0;
		GLuint m_Ibo = 0;
		GLuint m_Pbo = 0;
		GLuint m_Tex = 0;
		GLuint m_Prg = 0;

		GLint m_TexLoc = 0;
	};
}
#endif
