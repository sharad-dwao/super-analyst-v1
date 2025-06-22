"use client";

import type { Message } from "ai";
import { motion } from "framer-motion";

import { SparklesIcon } from "./icons";
import { Markdown } from "./markdown";
import { PreviewAttachment } from "./preview-attachment";
import { cn } from "@/lib/utils";

export const PreviewMessage = ({
  message,
}: {
  chatId: string;
  message: Message;
  isLoading: boolean;
}) => {
  return (
    <motion.div
      className="w-full mx-auto max-w-3xl px-4 group/message"
      initial={{ y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      data-role={message.role}
    >
      <div
        className={cn(
          "flex gap-4 w-full rounded-xl transition-all",
          message.role === "user" 
            ? "bg-primary text-primary-foreground px-3 py-2 ml-auto max-w-2xl w-fit" 
            : ""
        )}
      >
        {message.role === "assistant" && (
          <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border bg-background">
            <SparklesIcon size={14} />
          </div>
        )}

        <div className="flex flex-col gap-2 w-full min-w-0">
          {message.content && (
            <div className="flex flex-col gap-4">
              {message.role === "user" ? (
                <div className="text-primary-foreground font-medium">
                  {message.content as string}
                </div>
              ) : (
                <Markdown>{message.content as string}</Markdown>
              )}
            </div>
          )}

          {message.experimental_attachments && (
            <div className="flex flex-row gap-2 flex-wrap">
              {message.experimental_attachments.map((attachment) => (
                <PreviewAttachment
                  key={attachment.url}
                  attachment={attachment}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export const ThinkingMessage = () => {
  const role = "assistant";

  return (
    <motion.div
      className="w-full mx-auto max-w-3xl px-4 group/message"
      initial={{ y: 10, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: -10, opacity: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      data-role={role}
    >
      <div className="flex gap-4 w-full rounded-xl transition-all">
        <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border bg-background">
          <motion.div
            animate={{ 
              rotate: 360,
              scale: [1, 1.1, 1]
            }}
            transition={{ 
              rotate: { duration: 2, repeat: Infinity, ease: "linear" },
              scale: { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
            }}
          >
            <SparklesIcon size={14} />
          </motion.div>
        </div>

        <div className="flex flex-col gap-3 w-full min-w-0">
          <div className="flex items-center gap-3">
            <motion.span
              className="text-muted-foreground font-medium"
              animate={{ opacity: [0.6, 1, 0.6] }}
              transition={{ 
                duration: 2, 
                repeat: Infinity, 
                ease: "easeInOut" 
              }}
            >
              Analyzing your query
            </motion.span>
            
            {/* Animated dots */}
            <div className="flex gap-1">
              {[0, 1, 2].map((index) => (
                <motion.div
                  key={index}
                  className="w-1.5 h-1.5 bg-muted-foreground rounded-full"
                  animate={{
                    scale: [0.8, 1.2, 0.8],
                    opacity: [0.5, 1, 0.5]
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    delay: index * 0.2,
                    ease: "easeInOut"
                  }}
                />
              ))}
            </div>
          </div>
          
          {/* Progress bar animation */}
          <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-primary/60 via-primary to-primary/60 rounded-full"
              animate={{
                x: ["-100%", "100%"],
                opacity: [0.6, 1, 0.6]
              }}
              transition={{
                x: { duration: 2, repeat: Infinity, ease: "easeInOut" },
                opacity: { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
              }}
              style={{ width: "40%" }}
            />
          </div>
          
          {/* Subtle background pulse */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent rounded-xl -z-10"
            animate={{
              opacity: [0, 0.3, 0],
              scale: [0.98, 1.02, 0.98]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        </div>
      </div>
    </motion.div>
  );
};