{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.timer;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , ctypes
  ;

type

  { TSDLTimer }

  TSDLTimer = class
  private
    FID : TSDL_TimerID;
    FInterval : cuint32;
    FOnTimer: TNotifyEvent;
    function GetEnabled: Boolean;
    procedure SetInterval(AValue: cuint32);
    procedure SetOnTimer(AValue: TNotifyEvent);
  public
    constructor Create;
    destructor Destroy; override;
    procedure Start;
    procedure Stop;
    property Interval: cuint32 read FInterval write SetInterval;
    property OnTimer : TNotifyEvent read FOnTimer write SetOnTimer;
    property Enabled : Boolean read GetEnabled;
  end;

const
  SESSION_ONTIMER = SDL_USEREVENT+1;

implementation

{ TSDLTimer }

function TSDLTimer.GetEnabled: Boolean;
begin
  if FID > 0 then begin
    Result := True;
  end else begin
    Result := False;
  end;
end;

procedure TSDLTimer.SetInterval(AValue: cuint32);
begin
  if FInterval=AValue then Exit;
  FInterval:=AValue;
end;

function SDLOnTimer(interval: cuint32; param: Pointer): cuint32; cdecl;
var
  event : TSDL_Event;
  LTimer : TSDLTimer;
begin
  LTimer := TSDLTimer(param);
  if Assigned(LTimer) then
    if LTimer.Enabled then begin
      Result := interval;
    end else begin
      Result := 0;
    end;
  event.type_ := SESSION_ONTIMER;
  event.user.data1 := param;
  SDL_PushEvent(@event);
end;

procedure TSDLTimer.Start;
var
  Data : Pointer;
begin
  if Assigned(FOnTimer) then begin
    Stop;
    if Interval > 0 then begin
      Data := Pointer(Self);
      FID := SDL_AddTimer(FInterval, @SDLOnTimer, Data);
    end;
  end;
end;

procedure TSDLTimer.Stop;
begin
  if Enabled then begin
    if FID <> -1 then begin
      SDL_RemoveTimer(FID);
      FID := -1;
    end;
  end;
end;

procedure TSDLTimer.SetOnTimer(AValue: TNotifyEvent);
begin
  if FOnTimer=AValue then Exit;
  Stop;
  FOnTimer:=AValue;
end;

constructor TSDLTimer.Create;
begin
  inherited Create;
  FID := -1;
  FInterval := 0;
  FOnTimer := nil;
end;

destructor TSDLTimer.Destroy;
begin
  Stop;
  inherited Destroy;
end;

end.

