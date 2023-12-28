unit sdl.app.system.keyboard;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections, SDL2,
  sdl.app.events.abstract;

type

  TKeyDownEvents = specialize TList<TOnKeyDownEvent>;

  { TSDLSystemKeyboard }

  TSDLSystemKeyboard = class
  private
    FOnKeyDown : TOnKeyDownEvent;
    FOnKeyDownEvents : TKeyDownEvents;
    procedure KeyDown(const event: TSDL_KeyboardEvent);
  public
    constructor Create;
    destructor Destroy; override;
    procedure RegisterOnKeyDown(AOnKeyDownEvent : TOnKeyDownEvent);
    property OnKeyDown : TOnKeyDownEvent read FOnKeyDown;
  end;

implementation

uses sdl.app.events.custom, sdl.app;

{ TSDLKeyboard }

procedure TSDLSystemKeyboard.KeyDown(const event: TSDL_KeyboardEvent);
var
  LOnKeyDown : TOnKeyDownEvent;
begin
  case Event.keysym.sym of
    SDLK_ESCAPE: begin
      SDLApp.Terminate;
    end;

    otherwise begin
      for LOnKeyDown in FOnKeyDownEvents do begin
        LOnKeyDown(event);
      end;
    end;
  end;
end;

constructor TSDLSystemKeyboard.Create;
begin
  FOnKeyDownEvents := TKeyDownEvents.Create;
  FOnKeyDown:=@KeyDown;
end;

destructor TSDLSystemKeyboard.Destroy;
begin
  FOnKeyDownEvents.Free;
  inherited Destroy;
end;

procedure TSDLSystemKeyboard.RegisterOnKeyDown(
  AOnKeyDownEvent: TOnKeyDownEvent);
begin
  if Assigned(AOnKeyDownEvent) then begin
    FOnKeyDownEvents.Add(AOnKeyDownEvent);
  end;
end;

end.

end.

