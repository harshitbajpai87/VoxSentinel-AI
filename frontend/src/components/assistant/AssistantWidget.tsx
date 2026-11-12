import { useAssistant } from './AssistantProvider'
import { ChatWindow } from './ChatWindow'
import './assistant.css'

const AVATAR = '/assistant/avatar-192.png'

/**
 * The floating mascot plus the chat panel. Mounted once, above the router, so
 * it is present on every page and never re-created during navigation.
 */
export function AssistantWidget() {
  const { open, toggle, messages, pending, hasUnread, minimise } = useAssistant()
  const conversationStarted = messages.length > 0

  return (
    <>
      {open && <ChatWindow />}
      <button
        type="button"
        className={`assistant-launcher ${open ? 'is-open' : ''}`}
        onClick={() => (open ? minimise() : toggle())}
        aria-expanded={open}
        aria-label={open ? 'Minimise VoxSentinel Assistant' : 'Open VoxSentinel Assistant'}
        title="Need help? Ask VoxSentinel AI"
      >
        <span className="assistant-launcher-glow" aria-hidden="true" />
        <span className="assistant-launcher-float">
          <img src={AVATAR} alt="" className="assistant-launcher-img" />
          <span className="assistant-launcher-ping" aria-hidden="true" />
        </span>
        {(hasUnread || pending) && !open && (
          <span className={`assistant-badge ${pending ? 'is-busy' : ''}`} aria-hidden="true" />
        )}
        <span className="assistant-tooltip" role="tooltip">
          {conversationStarted ? 'Continue with VoxSentinel AI' : 'Need help? Ask VoxSentinel AI'}
        </span>
      </button>
    </>
  )
}
