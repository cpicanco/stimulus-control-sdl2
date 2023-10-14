{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.trials.dragdrop;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils
  , sdl.app.events.abstract
  , sdl.app.trials
  //, sdl.app.graphics.picture.dragdrop
  , sdl.app.stimuli.contract
  , sdl.app.stimuli.dragdrop
  , sdl.timer
  , session.configuration
  ;

type

  TReportData = record
    WrongDragDrops   : integer;
    Latency          : Extended;
    StimuliStart     : Extended;
  end;

  { TDragDrop }

  TDragDrop = class sealed (TTrial)
  private
    FUseHelpProgression : Boolean;
    FTimer : TSDLTimer;
    FReportData : TReportData;
    FStimuli : TDragDropStimuli;
    procedure DragDropDone(Sender : TObject);
    procedure TimerEndTrial(Sender: TObject);
  protected
    function GetIStimuli: IStimuli; override;
    procedure MouseMove(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
      override;
    procedure SetTrialData(ATrialData: TTrialData); override;
    procedure TrialLimitedHold(Sender: TObject);
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure EndTrial; override;
    procedure Show; override;
    procedure Hide; override;
  end;

implementation

uses
  session.constants.dragdrop
  , sdl.app.moveable.contract
  , sdl.app.renderer.custom
  ;

constructor TDragDrop.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);

  FTimer := TSDLTimer.Create;
  FTimer.Interval := 2000;
  FTimer.OnTimer := @TimerEndTrial;

  FStimuli := TDragDropStimuli.Create(Self);
  FStimuli.OnDragDropDone:=@DragDropDone;

  //FStimuli.LogEvent := @LogEvent;
  FReportData.WrongDragDrops := 0;
  FReportData.Latency := -1;

  FUseHelpProgression := False;
end;

destructor TDragDrop.Destroy;
begin
  FStimuli.Stop;
  FTimer.Free;
  inherited Destroy;
end;

procedure TDragDrop.EndTrial;
begin
  inherited EndTrial;
end;

procedure TDragDrop.Show;
begin
  inherited Show;
end;

procedure TDragDrop.Hide;
begin
  inherited Hide;
end;

procedure TDragDrop.TrialLimitedHold(Sender: TObject);
begin
  FStimuli.Stop;
end;

procedure TDragDrop.SetTrialData(ATrialData: TTrialData);
var
  LParameters: TStringList;
begin
  inherited SetTrialData(ATrialData);
  LParameters := ATrialData.Parameters;
  with DragDropKeys do
    FUseHelpProgression := LParameters.Values[UseHelpProgression].ToBoolean;

  //if FUseHelpProgression or FHasLimitedHold then begin
  //  if Counters.BlcTrials = 0 then begin
  //    IDragDropHelpSerie.AssignCurrent(LParameters);
  //  end;
  //  IDragDropHelpSerie.AssignParameters(LParameters);
  //end;
end;

procedure TDragDrop.DragDropDone(Sender: TObject);
begin
  FTimer.Start;
end;

procedure TDragDrop.TimerEndTrial(Sender: TObject);
begin
  EndTrial;
end;

function TDragDrop.GetIStimuli: IStimuli;
begin
  Result := FStimuli.AsInterface;
end;

procedure TDragDrop.MouseMove(Sender: TObject; Shift: TCustomShiftState; X,
  Y: Integer);
var
  Child : TComponent;
  IChild : IMoveable;
begin
  if Visible then begin
    for Child in FChilds do begin
      IChild := IMoveable(TCustomRenderer(Child));
      IChild.MouseMove(Sender, Shift, X, Y);
    end;
  end;
end;


end.

