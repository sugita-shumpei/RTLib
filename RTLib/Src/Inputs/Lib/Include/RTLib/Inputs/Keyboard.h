#ifndef RTLIB_INPUTS_KEY_BOARD_H
#define RTLIB_INPUTS_KEY_BOARD_H
#include <unordered_map>
namespace RTLib
{
    namespace Inputs
    {
        enum class KeyCode : unsigned int
        {
            eUnknown,
            eSpace,
            eApostrophe,
            eComma,
            eMinus,
            ePeriod,
            eSlash,
            e0,e1,e2,e3,e4,
            e5,e6,e7,e8,e9,
            eSemicolon,
            eEqual,
            eA,eB,eC,eD,
            eE,eF,eG,eH,
            eI,eJ,eK,eL,
            eM,eN,eO,eP,
            eQ,eR,eS,eT,
            eU,eV,eW,eX,
            eY,eZ,
            eLeftBracket,
            eBackSlash,
            eRightBracket,
            eGraveAccent,
            eWorld1,eWorld2,
            eEscape,
            eEnter,
            eTab,
            eBackSpace,
            eInsert, eDelete, 
            eRight, eLeft, eDown, eUp,
            ePageUp, ePageDown,
            eHome,eEnd,
            eCapsLock,eScrollLock,
            eNumLock,ePrintScreen,
            ePause,

            eF1,eF2,eF3,eF4,eF5,eF6,eF7,eF8,eF9,eF10,eF11,eF12,eF13,
            eF14,eF15,eF16,eF17,eF18,eF19,eF20,eF21,eF22,eF23,eF24,eF25,
            
            eKP0,eKP1,eKP2,eKP3,eKP4,
            eKP5,eKP6,eKP7,eKP8,eKP9,

            eKPDecimal, eKPDivide, eKPMultiply,eKPSubtract,
            eKPAdd,eKPEnter,eKPEqual,
            eLeftShift, eLeftControl, eLeftAlt,eLeftSuper,
            eRightShift,eRightControl,eRightAlt,eRightSuper,
            eMenu,

            eCount
        };
        enum KeyStateFlags : unsigned int
        {
            KeyStateReleased = 0,
            KeyStatePressed  = 1,
            KeyStateUpdated  = 2,
        };
        class Keyboard
        {
        public:
            using Callback = void(*)(KeyCode keyCode, unsigned int KeyState, void* pUserData);
        public:
            Keyboard() noexcept {}
            virtual ~Keyboard() noexcept{}

            auto GetKey(KeyCode keyCode)noexcept -> unsigned int {
                if (m_KeyStates.count(keyCode)==0){ return KeyStateReleased; }
                return m_KeyStates.at(keyCode);
            }

            void SetCallback(Callback callback)noexcept { m_Callback = callback;}
            auto GetCallback()const noexcept -> Callback{ return m_Callback;}

            void SetUserPointer(void* pUserData)noexcept{ m_PUserData = pUserData;}
            auto GetUserPointer()const noexcept -> void*{ return m_PUserData; }
        protected:
            auto Internal_KeyStates() const noexcept -> const std::unordered_map<KeyCode, unsigned int>&{
                return m_KeyStates;
            }
            auto Internal_KeyStates()       noexcept ->       std::unordered_map<KeyCode, unsigned int>&{
                return m_KeyStates;
            }
        private:
            static void DefaultCallback(KeyCode, unsigned int, void*){}
        private:
            std::unordered_map<KeyCode, unsigned int> m_KeyStates = {};
            Callback m_Callback = DefaultCallback;
            void* m_PUserData = nullptr;
        };
    }
}
#endif
