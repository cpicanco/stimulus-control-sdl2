{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.intertrial;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils
  , sdl.serialtimer
  ;

type

  { TInterTrialEvents }

  TInterTrialEvents = class(TComponent)
  private
    FSerialTimer : TSerialTimer;
    FInterTrial : TTimerItem;
    FDelay : TTimerItem;
    FConsequenceDuration : TTimerItem;
    procedure InterTrialConsequenceBegin;
    procedure InterTrialIntervalBegin;
    function HasDelay : Boolean;
    function HasConsequenceDuration : Boolean;
    function HasInterTrialTime : Boolean;
    procedure TrialEnd(Sender: TObject);
  private
    FOnEnd: TNotifyEvent;
    FOnBegin: TNotifyEvent;
    procedure InterTrialConsequenceEnd(Sender: TObject);
    procedure DelayEnd(Sender: TObject);
    procedure InterTrialEnd(Sender: TObject);
    procedure SetOnEnd(AValue: TNotifyEvent);
    procedure SetOnBegin(AValue: TNotifyEvent);
  public
    constructor Create(AOwner : TComponent); override;
    destructor Destroy; override;
    property OnEnd : TNotifyEvent read FOnEnd write SetOnEnd;
    property OnBegin : TNotifyEvent read FOnBegin write SetOnBegin;
  end;

var
  InterTrial : TInterTrialEvents;

implementation

uses
  session.loggers.writerow
  , session.pool
  , sdl.app.trials.contract
  , timestamps
  ;

{ TInterTrialEvents }

procedure TInterTrialEvents.SetOnBegin(AValue: TNotifyEvent);
begin
  if FOnBegin=AValue then Exit;
  FOnBegin:=AValue;
end;

procedure TInterTrialEvents.TrialEnd(Sender: TObject);
var
  LTrial : ITrial;
begin
  //ITIBegin := TickCount - Pool.TimeStart;
  LTrial := Sender as ITrial;
  //LTrial.Hide;

  //Background.Cursor := -1;
  FDelay.Interval := LTrial.ConsequenceDelay;
  FConsequenceDuration.Interval := LTrial.ConsequenceInterval;
  FInterTrial.Interval := LTrial.InterTrialInterval;

  if HasDelay then begin
    FSerialTimer.Append(FDelay);
  end;

  if HasConsequenceDuration then begin
    FSerialTimer.Append(FConsequenceDuration);
  end;

  if HasInterTrialTime then begin
    FSerialTimer.Append(FInterTrial);
  end;

  if HasDelay then begin
    FSerialTimer.Start;
    Exit;
  end;

  if HasConsequenceDuration then begin
    InterTrialConsequenceBegin;
    FSerialTimer.Start;
    Exit;
  end;

  if HasInterTrialTime then begin
    InterTrialIntervalBegin;
    FSerialTimer.Start;
    Exit;
  end;

  InterTrialEnd(Sender);
end;

function TInterTrialEvents.HasDelay: Boolean;
begin
  Result := FDelay.Interval > 0;
end;

function TInterTrialEvents.HasConsequenceDuration: Boolean;
begin
  Result := FConsequenceDuration.Interval > 0;
end;

function TInterTrialEvents.HasInterTrialTime: Boolean;
begin
  Result := FInterTrial.Interval > 0;
end;

procedure TInterTrialEvents.DelayEnd(Sender: TObject);
begin
  if HasConsequenceDuration then begin
    InterTrialConsequenceBegin;
    Exit;
  end;

  if HasInterTrialTime then begin

  end;
end;

procedure TInterTrialEvents.InterTrialConsequenceBegin;
begin
  //case TrialResult of
  //  'HIT+BLACKOUT':
  //    begin
  //
  //    end;
  //  'HIT' :
  //    begin
  //
  //    end;
  //
  //  'MISS':
  //    begin
  //
  //    end;
  //end;
  //TrialResult := '';
end;

procedure TInterTrialEvents.InterTrialConsequenceEnd(Sender: TObject);
begin
  if HasInterTrialTime then begin
    InterTrialIntervalBegin;
  end;
end;

procedure TInterTrialEvents.InterTrialIntervalBegin;
begin
  //ITIBegin := TickCount - Pool.TimeStart;
end;

procedure TInterTrialEvents.InterTrialEnd(Sender: TObject);
begin
  if Sender is TSerialTimer then begin
    FSerialTimer.Stop;
    FSerialTimer.Clear;
  end;

  FDelay.Interval := 0;
  FConsequenceDuration.Interval := 0;
  FInterTrial.Interval := 0;

  if Assigned(OnEnd) then
    OnEnd(Self);
end;

procedure TInterTrialEvents.SetOnEnd(AValue: TNotifyEvent);
begin
  if FOnEnd=AValue then Exit;
  FOnEnd:=AValue;
end;

constructor TInterTrialEvents.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FOnBegin := @TrialEnd;
  FSerialTimer := TSerialTimer.Create;

  FDelay.OnTimerEvent := @DelayEnd;
  FConsequenceDuration.OnTimerEvent := @InterTrialConsequenceEnd;
  FSerialTimer.OnEndTimeSerie := @InterTrialEnd;
end;

destructor TInterTrialEvents.Destroy;
begin
  FSerialTimer.Free;
  inherited Destroy;
end;

end.
