{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

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
    FReportData : TReportData;
    FStimuli : TDragDropStimuli;
  protected
    function GetIStimuli: IStimuli; override;
    procedure Reboot; override;
    procedure MouseMove(Sender: TObject;
      Shift: TCustomShiftState; X, Y: Integer); override;
  public
    constructor Create; override;
    destructor Destroy; override;
  end;

implementation

uses
  session.constants.trials.dragdrop
  , sdl.app.moveable.contract
  , sdl.app.controls.custom
  ;

constructor TDragDrop.Create;
begin
  inherited Create;

  FStimuli := TDragDropStimuli.Create;
  FStimuli.Trial := Self;
  FStimuli.OnFinalize:=@EndTrialCallBack;

  //FStimuli.LogEvent := @LogEvent;
  FReportData.WrongDragDrops := 0;
  FReportData.Latency := -1;
end;

destructor TDragDrop.Destroy;
begin
  FStimuli.Free;
  inherited Destroy;
end;

function TDragDrop.GetIStimuli: IStimuli;
begin
  Result := FStimuli.AsIStimuli;
end;

procedure TDragDrop.Reboot;
begin
  FStimuli.Reboot;
end;

procedure TDragDrop.MouseMove(Sender: TObject; Shift: TCustomShiftState; X,
  Y: Integer);
var
  Child : TObject;
  IChild : IMoveable;
begin
  if Visible then begin
    for Child in FChildren do begin
      IChild := IMoveable(TSDLControl(Child));
      IChild.MouseMove(Sender, Shift, X, Y);
    end;
  end;
end;


end.

