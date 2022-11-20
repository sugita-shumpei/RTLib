#include <RTLib/Inputs/Keyboard.h>
#include "Internals/GLFWInternals.h"
auto RTLib::Backends::Glfw::Internals::GetGlfwKeyCode(RTLib::Inputs::KeyCode code) -> int
{
    auto tmpCode = code;
    switch (tmpCode)
    {
    case RTLib::Inputs::KeyCode::eUnknown:
        return GLFW_KEY_UNKNOWN;
    case RTLib::Inputs::KeyCode::eSpace:
        return GLFW_KEY_SPACE;
    case RTLib::Inputs::KeyCode::eApostrophe:
        return GLFW_KEY_APOSTROPHE;
    case RTLib::Inputs::KeyCode::eComma:
        return GLFW_KEY_COMMA;
    case RTLib::Inputs::KeyCode::eMinus:
        return GLFW_KEY_MINUS;
    case RTLib::Inputs::KeyCode::ePeriod:
        return GLFW_KEY_PERIOD;
    case RTLib::Inputs::KeyCode::eSlash:
        return GLFW_KEY_SLASH;
    case RTLib::Inputs::KeyCode::e0:
        return GLFW_KEY_0;
    case RTLib::Inputs::KeyCode::e1:
        return GLFW_KEY_1;
    case RTLib::Inputs::KeyCode::e2:
        return GLFW_KEY_2;
    case RTLib::Inputs::KeyCode::e3:
        return GLFW_KEY_3;
    case RTLib::Inputs::KeyCode::e4:
        return GLFW_KEY_4;
    case RTLib::Inputs::KeyCode::e5:
        return GLFW_KEY_5;
    case RTLib::Inputs::KeyCode::e6:
        return GLFW_KEY_6;
    case RTLib::Inputs::KeyCode::e7:
        return GLFW_KEY_7;
    case RTLib::Inputs::KeyCode::e8:
        return GLFW_KEY_8;
    case RTLib::Inputs::KeyCode::e9:
        return GLFW_KEY_9;
    case RTLib::Inputs::KeyCode::eSemicolon:
        return GLFW_KEY_SEMICOLON;
    case RTLib::Inputs::KeyCode::eEqual:
        return GLFW_KEY_EQUAL;
    case RTLib::Inputs::KeyCode::eA:
        return GLFW_KEY_A;
    case RTLib::Inputs::KeyCode::eB:
        return GLFW_KEY_B;
    case RTLib::Inputs::KeyCode::eC:
        return GLFW_KEY_C;
    case RTLib::Inputs::KeyCode::eD:
        return GLFW_KEY_D;
    case RTLib::Inputs::KeyCode::eE:
        return GLFW_KEY_E;
    case RTLib::Inputs::KeyCode::eF:
        return GLFW_KEY_F;
    case RTLib::Inputs::KeyCode::eG:
        return GLFW_KEY_G;
    case RTLib::Inputs::KeyCode::eH:
        return GLFW_KEY_H;
    case RTLib::Inputs::KeyCode::eI:
        return GLFW_KEY_I;
    case RTLib::Inputs::KeyCode::eJ:
        return GLFW_KEY_J;
    case RTLib::Inputs::KeyCode::eK:
        return GLFW_KEY_K;
    case RTLib::Inputs::KeyCode::eL:
        return GLFW_KEY_L;
    case RTLib::Inputs::KeyCode::eM:
        return GLFW_KEY_M;
    case RTLib::Inputs::KeyCode::eN:
        return GLFW_KEY_N;
    case RTLib::Inputs::KeyCode::eO:
        return GLFW_KEY_O;
    case RTLib::Inputs::KeyCode::eP:
        return GLFW_KEY_P;
    case RTLib::Inputs::KeyCode::eQ:
        return GLFW_KEY_Q;
    case RTLib::Inputs::KeyCode::eR:
        return GLFW_KEY_R;
    case RTLib::Inputs::KeyCode::eS:
        return GLFW_KEY_S;
    case RTLib::Inputs::KeyCode::eT:
        return GLFW_KEY_T;
    case RTLib::Inputs::KeyCode::eU:
        return GLFW_KEY_U;
    case RTLib::Inputs::KeyCode::eV:
        return GLFW_KEY_V;
    case RTLib::Inputs::KeyCode::eW:
        return GLFW_KEY_W;
    case RTLib::Inputs::KeyCode::eX:
        return GLFW_KEY_X;
    case RTLib::Inputs::KeyCode::eY:
        return GLFW_KEY_Y;
    case RTLib::Inputs::KeyCode::eZ:
        return GLFW_KEY_Z;
    case RTLib::Inputs::KeyCode::eLeftBracket:
        return GLFW_KEY_LEFT_BRACKET;
    case RTLib::Inputs::KeyCode::eBackSlash:
        return GLFW_KEY_BACKSLASH;
    case RTLib::Inputs::KeyCode::eRightBracket:
        return GLFW_KEY_RIGHT_BRACKET;
    case RTLib::Inputs::KeyCode::eGraveAccent:
        return GLFW_KEY_GRAVE_ACCENT;
    case RTLib::Inputs::KeyCode::eWorld1:
        return GLFW_KEY_WORLD_1;
    case RTLib::Inputs::KeyCode::eWorld2:
        return GLFW_KEY_WORLD_2;
    case RTLib::Inputs::KeyCode::eEscape:
        return GLFW_KEY_ESCAPE;
    case RTLib::Inputs::KeyCode::eEnter:
        return GLFW_KEY_ENTER;
    case RTLib::Inputs::KeyCode::eTab:
        return GLFW_KEY_TAB;
    case RTLib::Inputs::KeyCode::eBackSpace:
        return GLFW_KEY_BACKSPACE;
    case RTLib::Inputs::KeyCode::eInsert:
        return GLFW_KEY_INSERT;
    case RTLib::Inputs::KeyCode::eDelete:
        return GLFW_KEY_DELETE;
    case RTLib::Inputs::KeyCode::eRight:
        return GLFW_KEY_RIGHT;
    case RTLib::Inputs::KeyCode::eLeft:
        return GLFW_KEY_LEFT;
    case RTLib::Inputs::KeyCode::eDown:
        return GLFW_KEY_DOWN;
    case RTLib::Inputs::KeyCode::eUp:
        return GLFW_KEY_UP;
    case RTLib::Inputs::KeyCode::ePageUp:
        return GLFW_KEY_PAGE_UP;
    case RTLib::Inputs::KeyCode::ePageDown:
        return GLFW_KEY_PAGE_DOWN;
    case RTLib::Inputs::KeyCode::eHome:
        return GLFW_KEY_HOME;
    case RTLib::Inputs::KeyCode::eEnd:
        return GLFW_KEY_END;
    case RTLib::Inputs::KeyCode::eCapsLock:
        return GLFW_KEY_CAPS_LOCK;
    case RTLib::Inputs::KeyCode::eScrollLock:
        return GLFW_KEY_SCROLL_LOCK;
    case RTLib::Inputs::KeyCode::eNumLock:
        return GLFW_KEY_NUM_LOCK;
    case RTLib::Inputs::KeyCode::ePrintScreen:
        return GLFW_KEY_PRINT_SCREEN;
    case RTLib::Inputs::KeyCode::ePause:
        return GLFW_KEY_PAUSE;
    case RTLib::Inputs::KeyCode::eF1:
        return GLFW_KEY_F1;
    case RTLib::Inputs::KeyCode::eF2:
        return GLFW_KEY_F2;
    case RTLib::Inputs::KeyCode::eF3:
        return GLFW_KEY_F3;
    case RTLib::Inputs::KeyCode::eF4:
        return GLFW_KEY_F4;
    case RTLib::Inputs::KeyCode::eF5:
        return GLFW_KEY_F5;
    case RTLib::Inputs::KeyCode::eF6:
        return GLFW_KEY_F6;
    case RTLib::Inputs::KeyCode::eF7:
        return GLFW_KEY_F7;
    case RTLib::Inputs::KeyCode::eF8:
        return GLFW_KEY_F8;
    case RTLib::Inputs::KeyCode::eF9:
        return GLFW_KEY_F9;
    case RTLib::Inputs::KeyCode::eF10:
        return GLFW_KEY_F10;
    case RTLib::Inputs::KeyCode::eF11:
        return GLFW_KEY_F11;
    case RTLib::Inputs::KeyCode::eF12:
        return GLFW_KEY_F12;
    case RTLib::Inputs::KeyCode::eF13:
        return GLFW_KEY_F13;
    case RTLib::Inputs::KeyCode::eF14:
        return GLFW_KEY_F14;
    case RTLib::Inputs::KeyCode::eF15:
        return GLFW_KEY_F15;
    case RTLib::Inputs::KeyCode::eF16:
        return GLFW_KEY_F16;
    case RTLib::Inputs::KeyCode::eF17:
        return GLFW_KEY_F17;
    case RTLib::Inputs::KeyCode::eF18:
        return GLFW_KEY_F18;
    case RTLib::Inputs::KeyCode::eF19:
        return GLFW_KEY_F19;
    case RTLib::Inputs::KeyCode::eF20:
        return GLFW_KEY_F20;
    case RTLib::Inputs::KeyCode::eF21:
        return GLFW_KEY_F21;
    case RTLib::Inputs::KeyCode::eF22:
        return GLFW_KEY_F22;
    case RTLib::Inputs::KeyCode::eF23:
        return GLFW_KEY_F23;
    case RTLib::Inputs::KeyCode::eF24:
        return GLFW_KEY_F24;
    case RTLib::Inputs::KeyCode::eF25:
        return GLFW_KEY_F25;
    case RTLib::Inputs::KeyCode::eKP0:
        return GLFW_KEY_KP_0;
    case RTLib::Inputs::KeyCode::eKP1:
        return GLFW_KEY_KP_1;
    case RTLib::Inputs::KeyCode::eKP2:
        return GLFW_KEY_KP_2;
    case RTLib::Inputs::KeyCode::eKP3:
        return GLFW_KEY_KP_3;
    case RTLib::Inputs::KeyCode::eKP4:
        return GLFW_KEY_KP_4;
    case RTLib::Inputs::KeyCode::eKP5:
        return GLFW_KEY_KP_5;
    case RTLib::Inputs::KeyCode::eKP6:
        return GLFW_KEY_KP_6;
    case RTLib::Inputs::KeyCode::eKP7:
        return GLFW_KEY_KP_7;
    case RTLib::Inputs::KeyCode::eKP8:
        return GLFW_KEY_KP_8;
    case RTLib::Inputs::KeyCode::eKP9:
        return GLFW_KEY_KP_9;
    case RTLib::Inputs::KeyCode::eKPDecimal:
        return GLFW_KEY_KP_DECIMAL;
    case RTLib::Inputs::KeyCode::eKPDivide:
        return GLFW_KEY_KP_DIVIDE;
    case RTLib::Inputs::KeyCode::eKPMultiply:
        return GLFW_KEY_KP_MULTIPLY;
    case RTLib::Inputs::KeyCode::eKPSubtract:
        return GLFW_KEY_KP_SUBTRACT;
    case RTLib::Inputs::KeyCode::eKPAdd:
        return GLFW_KEY_KP_ADD;
    case RTLib::Inputs::KeyCode::eKPEnter:
        return GLFW_KEY_KP_ENTER;
    case RTLib::Inputs::KeyCode::eKPEqual:
        return GLFW_KEY_KP_EQUAL;
    case RTLib::Inputs::KeyCode::eLeftShift:
        return GLFW_KEY_LEFT_SHIFT;
    case RTLib::Inputs::KeyCode::eLeftControl:
        return GLFW_KEY_LEFT_CONTROL;
    case RTLib::Inputs::KeyCode::eLeftAlt:
        return GLFW_KEY_LEFT_ALT;
    case RTLib::Inputs::KeyCode::eLeftSuper:
        return GLFW_KEY_LEFT_SUPER;
    case RTLib::Inputs::KeyCode::eRightShift:
        return GLFW_KEY_RIGHT_SHIFT;
    case RTLib::Inputs::KeyCode::eRightControl:
        return GLFW_KEY_RIGHT_CONTROL;
    case RTLib::Inputs::KeyCode::eRightAlt:
        return GLFW_KEY_RIGHT_ALT;
    case RTLib::Inputs::KeyCode::eRightSuper:
        return GLFW_KEY_RIGHT_SUPER;
    case RTLib::Inputs::KeyCode::eMenu:
        return GLFW_KEY_MENU;
    case RTLib::Inputs::KeyCode::eCount:
        return GLFW_KEY_UNKNOWN;
    default:
        return GLFW_KEY_UNKNOWN;
    }
}
auto RTLib::Backends::Glfw::Internals::GetInptKeyCode(int keyCode) -> RTLib::Inputs::KeyCode
{
    switch (keyCode)
    {
    case GLFW_KEY_UNKNOWN:
        return RTLib::Inputs::KeyCode::eUnknown;
    case GLFW_KEY_SPACE:
        return RTLib::Inputs::KeyCode::eSpace;
    case GLFW_KEY_APOSTROPHE:
        return RTLib::Inputs::KeyCode::eApostrophe;
    case GLFW_KEY_COMMA:
        return RTLib::Inputs::KeyCode::eComma;
    case GLFW_KEY_MINUS:
        return RTLib::Inputs::KeyCode::eMinus;
    case GLFW_KEY_PERIOD:
        return RTLib::Inputs::KeyCode::ePeriod;
    case GLFW_KEY_SLASH:
        return RTLib::Inputs::KeyCode::eSlash;
    case GLFW_KEY_0:
        return RTLib::Inputs::KeyCode::e0;
    case GLFW_KEY_1:
        return RTLib::Inputs::KeyCode::e1;
    case GLFW_KEY_2:
        return RTLib::Inputs::KeyCode::e2;
    case GLFW_KEY_3:
        return RTLib::Inputs::KeyCode::e3;
    case GLFW_KEY_4:
        return RTLib::Inputs::KeyCode::e4;
    case GLFW_KEY_5:
        return RTLib::Inputs::KeyCode::e5;
    case GLFW_KEY_6:
        return RTLib::Inputs::KeyCode::e6;
    case GLFW_KEY_7:
        return RTLib::Inputs::KeyCode::e7;
    case GLFW_KEY_8:
        return RTLib::Inputs::KeyCode::e8;
    case GLFW_KEY_9:
        return RTLib::Inputs::KeyCode::e9;
    case GLFW_KEY_SEMICOLON:
        return RTLib::Inputs::KeyCode::eSemicolon;
    case GLFW_KEY_EQUAL:
        return RTLib::Inputs::KeyCode::eEqual;
    case GLFW_KEY_A:
        return RTLib::Inputs::KeyCode::eA;
    case GLFW_KEY_B:
        return RTLib::Inputs::KeyCode::eB;
    case GLFW_KEY_C:
        return RTLib::Inputs::KeyCode::eC;
    case GLFW_KEY_D:
        return RTLib::Inputs::KeyCode::eD;
    case GLFW_KEY_E:
        return RTLib::Inputs::KeyCode::eE;
    case GLFW_KEY_F:
        return RTLib::Inputs::KeyCode::eF;
    case GLFW_KEY_G:
        return RTLib::Inputs::KeyCode::eG;
    case GLFW_KEY_H:
        return RTLib::Inputs::KeyCode::eH;
    case GLFW_KEY_I:
        return RTLib::Inputs::KeyCode::eI;
    case GLFW_KEY_J:
        return RTLib::Inputs::KeyCode::eJ;
    case GLFW_KEY_K:
        return RTLib::Inputs::KeyCode::eK;
    case GLFW_KEY_L:
        return RTLib::Inputs::KeyCode::eL;
    case GLFW_KEY_M:
        return RTLib::Inputs::KeyCode::eM;
    case GLFW_KEY_N:
        return RTLib::Inputs::KeyCode::eN;
    case GLFW_KEY_O:
        return RTLib::Inputs::KeyCode::eO;
    case GLFW_KEY_P:
        return RTLib::Inputs::KeyCode::eP;
    case GLFW_KEY_Q:
        return RTLib::Inputs::KeyCode::eQ;
    case GLFW_KEY_R:
        return RTLib::Inputs::KeyCode::eR;
    case GLFW_KEY_S:
        return RTLib::Inputs::KeyCode::eS;
    case GLFW_KEY_T:
        return RTLib::Inputs::KeyCode::eT;
    case GLFW_KEY_U:
        return RTLib::Inputs::KeyCode::eU;
    case GLFW_KEY_V:
        return RTLib::Inputs::KeyCode::eV;
    case GLFW_KEY_W:
        return RTLib::Inputs::KeyCode::eW;
    case GLFW_KEY_X:
        return RTLib::Inputs::KeyCode::eX;
    case GLFW_KEY_Y:
        return RTLib::Inputs::KeyCode::eY;
    case GLFW_KEY_Z:
        return RTLib::Inputs::KeyCode::eZ;
    case GLFW_KEY_LEFT_BRACKET:
        return RTLib::Inputs::KeyCode::eLeftBracket;
    case GLFW_KEY_BACKSLASH:
        return RTLib::Inputs::KeyCode::eBackSlash;
    case GLFW_KEY_RIGHT_BRACKET:
        return RTLib::Inputs::KeyCode::eRightBracket;
    case GLFW_KEY_GRAVE_ACCENT:
        return RTLib::Inputs::KeyCode::eGraveAccent;
    case GLFW_KEY_WORLD_1:
        return RTLib::Inputs::KeyCode::eWorld1;
    case GLFW_KEY_WORLD_2:
        return RTLib::Inputs::KeyCode::eWorld2;
    case GLFW_KEY_ESCAPE:
        return RTLib::Inputs::KeyCode::eEscape;
    case GLFW_KEY_ENTER:
        return RTLib::Inputs::KeyCode::eEnter;
    case GLFW_KEY_TAB:
        return RTLib::Inputs::KeyCode::eTab;
    case GLFW_KEY_BACKSPACE:
        return RTLib::Inputs::KeyCode::eBackSpace;
    case GLFW_KEY_INSERT:
        return RTLib::Inputs::KeyCode::eInsert;
    case GLFW_KEY_DELETE:
        return RTLib::Inputs::KeyCode::eDelete;
    case GLFW_KEY_RIGHT:
        return RTLib::Inputs::KeyCode::eRight;
    case GLFW_KEY_LEFT:
        return RTLib::Inputs::KeyCode::eLeft;
    case GLFW_KEY_DOWN:
        return RTLib::Inputs::KeyCode::eDown;
    case GLFW_KEY_UP:
        return RTLib::Inputs::KeyCode::eUp;
    case GLFW_KEY_PAGE_UP:
        return RTLib::Inputs::KeyCode::ePageUp;
    case GLFW_KEY_PAGE_DOWN:
        return RTLib::Inputs::KeyCode::ePageDown;
    case GLFW_KEY_HOME:
        return RTLib::Inputs::KeyCode::eHome;
    case GLFW_KEY_END:
        return RTLib::Inputs::KeyCode::eEnd;
    case GLFW_KEY_CAPS_LOCK:
        return RTLib::Inputs::KeyCode::eCapsLock;
    case GLFW_KEY_SCROLL_LOCK:
        return RTLib::Inputs::KeyCode::eScrollLock;
    case GLFW_KEY_NUM_LOCK:
        return RTLib::Inputs::KeyCode::eNumLock;
    case GLFW_KEY_PRINT_SCREEN:
        return RTLib::Inputs::KeyCode::ePrintScreen;
    case GLFW_KEY_PAUSE:
        return RTLib::Inputs::KeyCode::ePause;
    case GLFW_KEY_F1:
        return RTLib::Inputs::KeyCode::eF1;
    case GLFW_KEY_F2:
        return RTLib::Inputs::KeyCode::eF2;
    case GLFW_KEY_F3:
        return RTLib::Inputs::KeyCode::eF3;
    case GLFW_KEY_F4:
        return RTLib::Inputs::KeyCode::eF4;
    case GLFW_KEY_F5:
        return RTLib::Inputs::KeyCode::eF5;
    case GLFW_KEY_F6:
        return RTLib::Inputs::KeyCode::eF6;
    case GLFW_KEY_F7:
        return RTLib::Inputs::KeyCode::eF7;
    case GLFW_KEY_F8:
        return RTLib::Inputs::KeyCode::eF8;
    case GLFW_KEY_F9:
        return RTLib::Inputs::KeyCode::eF9;
    case GLFW_KEY_F10:
        return RTLib::Inputs::KeyCode::eF10;
    case GLFW_KEY_F11:
        return RTLib::Inputs::KeyCode::eF11;
    case GLFW_KEY_F12:
        return RTLib::Inputs::KeyCode::eF12;
    case GLFW_KEY_F13:
        return RTLib::Inputs::KeyCode::eF13;
    case GLFW_KEY_F14:
        return RTLib::Inputs::KeyCode::eF14;
    case GLFW_KEY_F15:
        return RTLib::Inputs::KeyCode::eF15;
    case GLFW_KEY_F16:
        return RTLib::Inputs::KeyCode::eF16;
    case GLFW_KEY_F17:
        return RTLib::Inputs::KeyCode::eF17;
    case GLFW_KEY_F18:
        return RTLib::Inputs::KeyCode::eF18;
    case GLFW_KEY_F19:
        return RTLib::Inputs::KeyCode::eF19;
    case GLFW_KEY_F20:
        return RTLib::Inputs::KeyCode::eF20;
    case GLFW_KEY_F21:
        return RTLib::Inputs::KeyCode::eF21;
    case GLFW_KEY_F22:
        return RTLib::Inputs::KeyCode::eF22;
    case GLFW_KEY_F23:
        return RTLib::Inputs::KeyCode::eF23;
    case GLFW_KEY_F24:
        return RTLib::Inputs::KeyCode::eF24;
    case GLFW_KEY_F25:
        return RTLib::Inputs::KeyCode::eF25;
    case GLFW_KEY_KP_0:
        return RTLib::Inputs::KeyCode::eKP0;
    case GLFW_KEY_KP_1:
        return RTLib::Inputs::KeyCode::eKP1;
    case GLFW_KEY_KP_2:
        return RTLib::Inputs::KeyCode::eKP2;
    case GLFW_KEY_KP_3:
        return RTLib::Inputs::KeyCode::eKP3;
    case GLFW_KEY_KP_4:
        return RTLib::Inputs::KeyCode::eKP4;
    case GLFW_KEY_KP_5:
        return RTLib::Inputs::KeyCode::eKP5;
    case GLFW_KEY_KP_6:
        return RTLib::Inputs::KeyCode::eKP6;
    case GLFW_KEY_KP_7:
        return RTLib::Inputs::KeyCode::eKP7;
    case GLFW_KEY_KP_8:
        return RTLib::Inputs::KeyCode::eKP8;
    case GLFW_KEY_KP_9:
        return RTLib::Inputs::KeyCode::eKP9;
    case GLFW_KEY_KP_DECIMAL:
        return RTLib::Inputs::KeyCode::eKPDecimal;
    case GLFW_KEY_KP_DIVIDE:
        return RTLib::Inputs::KeyCode::eKPDivide;
    case GLFW_KEY_KP_MULTIPLY:
        return RTLib::Inputs::KeyCode::eKPMultiply;
    case GLFW_KEY_KP_SUBTRACT:
        return RTLib::Inputs::KeyCode::eKPSubtract;
    case GLFW_KEY_KP_ADD:
        return RTLib::Inputs::KeyCode::eKPAdd;
    case GLFW_KEY_KP_ENTER:
        return RTLib::Inputs::KeyCode::eKPEnter;
    case GLFW_KEY_KP_EQUAL:
        return RTLib::Inputs::KeyCode::eKPEqual;
    case GLFW_KEY_LEFT_SHIFT:
        return RTLib::Inputs::KeyCode::eLeftShift;
    case GLFW_KEY_LEFT_CONTROL:
        return RTLib::Inputs::KeyCode::eLeftControl;
    case GLFW_KEY_LEFT_ALT:
        return RTLib::Inputs::KeyCode::eLeftAlt;
    case GLFW_KEY_LEFT_SUPER:
        return RTLib::Inputs::KeyCode::eLeftSuper;
    case GLFW_KEY_RIGHT_SHIFT:
        return RTLib::Inputs::KeyCode::eRightShift;
    case GLFW_KEY_RIGHT_CONTROL:
        return RTLib::Inputs::KeyCode::eRightControl;
    case GLFW_KEY_RIGHT_ALT:
        return RTLib::Inputs::KeyCode::eRightAlt;
    case GLFW_KEY_RIGHT_SUPER:
        return RTLib::Inputs::KeyCode::eRightSuper;
    case GLFW_KEY_MENU:
        return RTLib::Inputs::KeyCode::eMenu;
    default:
        return RTLib::Inputs::KeyCode::eUnknown;
    }
}