// Shared motion primitives for foreground animations only.
// The page background, its colors and any background decorations are NEVER
// animated or changed by these components.
//
// All entrances use transform/opacity only, fire once, and respect
// prefers-reduced-motion (motionAmount scales to 0 when the user asks for
// reduced motion, so elements simply appear).
import { motion, useReducedMotion, type Variants } from 'framer-motion'
import type { ReactNode } from 'react'

export function useMotionAmount() {
  return useReducedMotion() ? 0 : 1
}

/** Fade + gentle rise, optionally triggered when scrolled into view. */
export function Reveal({
  children,
  delay = 0,
  inView = false,
  className,
}: {
  children: ReactNode
  delay?: number
  inView?: boolean
  className?: string
}) {
  const amount = useMotionAmount()
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 12 * amount }}
      {...(inView
        ? { whileInView: { opacity: 1, y: 0 }, viewport: { once: true, margin: '-40px' } }
        : { animate: { opacity: 1, y: 0 } })}
      transition={{ duration: 0.45, delay, ease: [0.21, 0.6, 0.35, 1] }}
    >
      {children}
    </motion.div>
  )
}

/** Staggered container: children with `staggerIndex` reveal in sequence. */
export const staggerContainer: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07, delayChildren: 0.05 } },
}

export const staggerChild: Variants = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.21, 0.6, 0.35, 1] } },
}

export function StaggerGroup({ children, className }: { children: ReactNode; className?: string }) {
  const reduced = useReducedMotion()
  if (reduced) return <div className={className}>{children}</div>
  return (
    <motion.div
      className={className}
      variants={staggerContainer}
      initial="hidden"
      animate="show"
    >
      {children}
    </motion.div>
  )
}

export function StaggerItem({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div className={className} variants={staggerChild}>
      {children}
    </motion.div>
  )
}
