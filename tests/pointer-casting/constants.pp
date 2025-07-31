{$MACRO ON}
{$ifndef EXTERNAL_DEV_NONE}
  {$DEFINE EXTERNAL_DEV_NONE := TGetExButtonStatesFunction(Pointer(0))}
{$endif}
{$ifndef EXTERNAL_DEV_CEDRUS}
  {$DEFINE EXTERNAL_DEV_CEDRUS := TGetExButtonStatesFunction(Pointer(1))}
{$endif}
{$ifndef EXTERNAL_DEV_SYS_KEYBOARD}
  {$DEFINE EXTERNAL_DEV_SYS_KEYBOARD := TGetExButtonStatesFunction(Pointer(2))}
{$endif}
