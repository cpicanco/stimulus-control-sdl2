{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.serialtimer;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils
  , ctypes
  , sdl.timer
  ;

type

  TTimerItem = record
    Interval : cint32;
    OnTimerEvent : TNotifyEvent;
  end;

  TTimerItems = array of TTimerItem;

  { TSerialTimer }

  TSerialTimer = class
  private
    FOnEndTimeSerie: TNotifyEvent;
    FTimer : TSDLTimer;
    FCurrentData : TTimerItem;
    FCurrentItem : Cardinal;
    FTimerItems : TTimerItems;
    function GetCurrentTimeItem: TTimerItem;
    function GetCurrentTimerItemPointer: Pointer;
    procedure SetCurrentItem(AValue: Cardinal);
    procedure SetOnEndTimeSerie(AValue: TNotifyEvent);
    procedure TimerOnTimer(Sender : TObject);
  public
    constructor Create; reintroduce;
    destructor Destroy; override;
    procedure Append(ATimerItem : TTimerItem); overload;
    procedure Append(ATimerItems : TTimerItems); overload;
    procedure Clear;
    procedure Start;
    procedure Stop;
    procedure Restart;
    function Interval(AItem : cint32) : cint32;
    function HasNextItem : Boolean;
    procedure NextItem;
    property CurrentItem : Cardinal read FCurrentItem write SetCurrentItem;
    property CurrentTimerItem : TTimerItem read GetCurrentTimeItem;
    property CurrentTimerItemPointer : Pointer read GetCurrentTimerItemPointer;
    property OnEndTimeSerie : TNotifyEvent read FOnEndTimeSerie write SetOnEndTimeSerie;
  end;

implementation

//uses sdl.app.output;

{ TSerialTimer }

procedure TSerialTimer.SetCurrentItem(AValue: Cardinal);
begin
  if (FCurrentItem > High(FTimerItems)) or
     (FCurrentItem = AValue) then Exit;
  if FTimer.Enabled then Stop;
  FCurrentItem := AValue;
end;

function TSerialTimer.GetCurrentTimeItem: TTimerItem;
begin
  FCurrentData := FTimerItems[CurrentItem];
  Result := FCurrentData;
end;

function TSerialTimer.GetCurrentTimerItemPointer: Pointer;
begin
  Result := @FCurrentData;
end;

procedure TSerialTimer.SetOnEndTimeSerie(AValue: TNotifyEvent);
begin
  if FOnEndTimeSerie = AValue then Exit;
  FOnEndTimeSerie := AValue;
end;

procedure TSerialTimer.TimerOnTimer(Sender: TObject);
var
  LTimerItem: TTimerItem;
begin
  LTimerItem := GetCurrentTimeItem;
  if Assigned(LTimerItem.OnTimerEvent) then begin
    LTimerItem.OnTimerEvent(Self);
  end;

  if HasNextItem then begin
    NextItem;
    LTimerItem := GetCurrentTimeItem;
    if LTimerItem.Interval = 0 then begin
      TimerOnTimer(Sender);
    end;

  end else begin
    if Assigned(OnEndTimeSerie) then
      OnEndTimeSerie(Self);
  end;
end;

constructor TSerialTimer.Create;
begin
  FCurrentItem := 0;
  FTimer := TSDLTimer.Create;
  FTimer.OnTimer := @TimerOnTimer;
end;

destructor TSerialTimer.Destroy;
begin
  Clear;
  FTimer.Free;
  inherited Destroy;
end;

procedure TSerialTimer.Append(ATimerItem: TTimerItem);
begin
  SetLength(FTimerItems, Length(FTimerItems)+1);
  FTimerItems[High(FTimerItems)] := ATimerItem;
end;

procedure TSerialTimer.Append(ATimerItems: TTimerItems);
var
  LTimerItem : TTimerItem;
begin
  if Length(ATimerItems) = 0 then Exit;
  for LTimerItem in ATimerItems do
    Append(LTimerItem);
end;

procedure TSerialTimer.Clear;
begin
  if FTimer.Enabled then Stop;
  SetLength(FTimerItems, 0);
  FCurrentItem := 0;
  FCurrentData.OnTimerEvent := nil;
  FCurrentData.Interval := 0;
end;

procedure TSerialTimer.Start;
begin
  if Length(FTimerItems) = 0 then Exit;
  FCurrentData := GetCurrentTimeItem;
  FTimer.Interval := FTimerItems[FCurrentItem].Interval;
  FTimer.Start;
end;

procedure TSerialTimer.Stop;
begin
  FTimer.Stop;
end;

procedure TSerialTimer.Restart;
begin
  if FTimer.Enabled then Stop;
  FCurrentItem := 0;
  FCurrentData := GetCurrentTimeItem;
  Start;
end;

function TSerialTimer.Interval(AItem: cint32): cint32;
begin
  Result := FTimerItems[AItem].Interval;
end;

function TSerialTimer.HasNextItem: Boolean;
begin
  Result := FCurrentItem < High(FTimerItems);
end;

procedure TSerialTimer.NextItem;
begin
  Inc(FCurrentItem);
  FCurrentData := GetCurrentTimeItem;
end;


end.
