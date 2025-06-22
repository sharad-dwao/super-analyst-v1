import type { Attachment } from "ai";

import { LoaderIcon } from "./icons";

export const PreviewAttachment = ({
  attachment,
  isUploading = false,
}: {
  attachment: Attachment;
  isUploading?: boolean;
}) => {
  const { name, url, contentType } = attachment;

  return (
    <div className="flex flex-col gap-2 flex-shrink-0">
      <div className="w-20 aspect-video bg-muted rounded-md relative flex flex-col items-center justify-center border border-border overflow-hidden">
        {contentType ? (
          contentType.startsWith("image") ? (
            // NOTE: it is recommended to use next/image for images
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={url}
              src={url}
              alt={name ?? "An image attachment"}
              className="rounded-md size-full object-cover"
            />
          ) : (
            <div className="text-muted-foreground text-xs">File</div>
          )
        ) : (
          <div className="text-muted-foreground text-xs">Unknown</div>
        )}

        {isUploading && (
          <div className="animate-spin absolute text-muted-foreground">
            <LoaderIcon />
          </div>
        )}
      </div>
      <div className="text-xs text-muted-foreground max-w-20 truncate text-center">
        {name}
      </div>
    </div>
  );
};