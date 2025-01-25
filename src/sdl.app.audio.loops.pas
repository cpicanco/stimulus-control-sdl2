{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.audio.loops;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils,
  sdl.app.audio.contract,
  sdl.timer,
  ctypes;

type

  { TSoundLoop }

  TSoundLoop = class
    private
      FOnEveryLoopStart: TNotifyEvent;
      FOnEveryLoopStop: TNotifyEvent;
      FOnFinalLoopStop: TNotifyEvent;
      FTotalLoops : SmallInt;
      FCurrentLoopCount : SmallInt;
      FSound : ISound;
      FTimer : TSDLTimer;
      function GetInterval: cuint32;
      function IsFinalLoop : Boolean;
      procedure SetInterval(AValue: cuint32);
      procedure PlaySound(Sender: TObject);
      procedure SetOnEveryLoopStart(AValue: TNotifyEvent);
      procedure SetOnEveryLoopStop(AValue: TNotifyEvent);
      procedure SetOnFinalLoopStop(AValue: TNotifyEvent);
      procedure EveryLoopStart(Sender: TObject);
      procedure EveryLoopStop(Sender: TObject);
      procedure SetTotalLoops(AValue: SmallInt);
      procedure Validate;
    public
      constructor Create;
      destructor Destroy; override;
      property Sound : ISound read FSound write FSound;
      property Interval : cuint32 read GetInterval write SetInterval;
      property TotalLoops : SmallInt read FTotalLoops write SetTotalLoops;
      procedure Start;
      procedure Stop;
      property OnEveryLoopStart: TNotifyEvent read FOnEveryLoopStart write SetOnEveryLoopStart;
      property OnEveryLoopStop : TNotifyEvent read FOnEveryLoopStop write SetOnEveryLoopStop;
      property OnFinalLoopStop : TNotifyEvent read FOnFinalLoopStop write SetOnFinalLoopStop;
  end;

implementation

uses session.parameters.global;

{ TSoundLoop }

function TSoundLoop.GetInterval: cuint32;
begin
  Result := FTimer.Interval;
end;

function TSoundLoop.IsFinalLoop: Boolean;
begin
  Result := FCurrentLoopCount >= FTotalLoops;
end;

procedure TSoundLoop.SetInterval(AValue: cuint32);
begin
  FTimer.Interval := AValue;
end;

procedure TSoundLoop.PlaySound(Sender: TObject);
begin
  if IsFinalLoop then begin
    FTimer.Stop;
    FCurrentLoopCount := 0;
    if Assigned(OnFinalLoopStop) then begin
      OnFinalLoopStop(Sender);
    end;
    Exit;
  end;
  FSound.Play;
end;

procedure TSoundLoop.SetOnEveryLoopStart(AValue: TNotifyEvent);
begin
  Validate;
  if FOnEveryLoopStart = AValue then Exit;
  FOnEveryLoopStart := AValue;
  FSound.SetOnStart(@EveryLoopStart);
end;

procedure TSoundLoop.SetOnEveryLoopStop(AValue: TNotifyEvent);
begin
  Validate;
  if FOnEveryLoopStop = AValue then Exit;
  FOnEveryLoopStop := AValue;
  FSound.SetOnStop(@EveryLoopStop);
end;

procedure TSoundLoop.SetOnFinalLoopStop(AValue: TNotifyEvent);
begin
  if FOnFinalLoopStop = AValue then Exit;
  FOnFinalLoopStop := AValue;
end;

procedure TSoundLoop.EveryLoopStart(Sender: TObject);
begin
  if FSound = nil then Exit;
  Inc(FCurrentLoopCount);
  if Assigned(OnEveryLoopStart) then begin
    OnEveryLoopStart(Sender);
  end;
end;

procedure TSoundLoop.EveryLoopStop(Sender: TObject);
begin
  if FSound = nil then Exit;
  if Assigned(OnEveryLoopStop) then begin
    OnEveryLoopStop(Sender);
  end;
end;

procedure TSoundLoop.SetTotalLoops(AValue: SmallInt);
begin
  if FTotalLoops = AValue then Exit;
  if AValue < 0 then begin
    FTotalLoops := MaxSmallint;
  end else begin
    FTotalLoops := AValue;
  end;
end;

procedure TSoundLoop.Validate;
begin
  if FSound = nil then raise
    EInvalidCast.Create('You must assign TSoundLoop.Sound first.');
end;

constructor TSoundLoop.Create;
begin
  FSound := nil;
  FTimer := TSDLTimer.Create;
  FTimer.OnTimer := @PlaySound;
  FTimer.Interval := GlobalTrialParameters.AudioLoopInterval;
  FCurrentLoopCount := 0;
  FTotalLoops := 0;
end;

destructor TSoundLoop.Destroy;
begin
  FTimer.OnTimer := nil;
  FTimer.Free;
  inherited Destroy;
end;

procedure TSoundLoop.Start;
begin
  Validate;
  case FTotalLoops of
    1 : begin
      if not FSound.Playing then begin
        FSound.Play;
      end;
    end;

    2..MaxSmallint: begin
      if not FSound.Playing then begin
        FSound.Play;
      end;

      if not FTimer.Enabled then begin
        FTimer.Start;
      end;
    end;

    else Exit;
  end;
end;

procedure TSoundLoop.Stop;
begin
  Validate;
  FTimer.Stop;
  FSound.Stop;
end;

end.

